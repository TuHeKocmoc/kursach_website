from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from app.ml.features import (
    clean_ohlcv,
    clipped_log_return,
    latest_feature_vector,
    latest_lstm_sequence,
    make_lstm_dataset,
    make_supervised_dataset,
)
from app.ml.pdt import PermutationDecisionTreeRegressor

ForecastModelName = Literal["pdt", "lstm", "xgb", "naive"]


@dataclass(frozen=True)
class MetricResult:
    model: str
    horizon_days: int
    train_size: int
    test_size: int
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    naive_rmse: float
    naive_mae: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastResult:
    values: list[float]
    metrics: MetricResult | None
    warnings: list[str]
    model_display_name: str


class ModelTrainingError(RuntimeError):
    pass


def _model_display_name(model: str) -> str:
    return {
        "pdt": "Permutation Decision Tree",
        "lstm": "LSTM",
        "xgb": "XGBoost",
        "naive": "Naive last-price baseline",
    }.get(model, model)


def _impute_with_train_medians(
    X_train: np.ndarray,
    X_other: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    med = np.nanmedian(np.where(np.isfinite(X_train), X_train, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    X_train_out = np.where(np.isfinite(X_train), X_train, med)

    if X_other is None:
        return X_train_out, None

    X_other_out = np.where(np.isfinite(X_other), X_other, med)
    return X_train_out, X_other_out


def _standardize_sequences(
    X_train: np.ndarray,
    X_other: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])

    mean = np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=0)
    std = np.nanstd(np.where(np.isfinite(flat), flat, np.nan), axis=0)

    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)

    X_train_s = np.nan_to_num(
        (X_train - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if X_other is None:
        return X_train_s, None, mean, std

    X_other_s = np.nan_to_num(
        (X_other - mean) / std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X_train_s, X_other_s, mean, std


def _make_pdt() -> PermutationDecisionTreeRegressor:
    return PermutationDecisionTreeRegressor(
        max_depth=5,
        min_samples_split=28,
        min_samples_leaf=12,
        n_thresholds=18,
        permutation_order=3,
        order_weight=0.30,
        random_state=42,
    )


def _make_xgb_regressor() -> object:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ModelTrainingError(
            "XGBoost model requires the xgboost package. Install project dependencies or choose another model."
        ) from exc

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def _fit_predict_tabular(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
) -> np.ndarray:
    X_train, X_pred = _impute_with_train_medians(X_train, X_pred)
    assert X_pred is not None

    if model == "pdt":
        reg = _make_pdt()
    elif model == "xgb":
        reg = _make_xgb_regressor()
    else:
        raise ValueError(f"Unsupported tabular model: {model}")

    reg.fit(X_train, y_train)
    return np.asarray(reg.predict(X_pred), dtype=float).reshape(-1)


def _drift_forecast(
    df: pd.DataFrame,
    horizon_days: int,
    zero_return: bool = False,
) -> list[float]:
    clean = clean_ohlcv(df)

    if clean.empty:
        raise ModelTrainingError("No prices for fallback forecast")

    close = clean["Close"].astype(float)
    last_close = float(close.iloc[-1])

    if zero_return:
        mu = 0.0
    else:
        returns = np.log(close / close.shift(1))
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        mu = float(returns.tail(30).mean()) if not returns.empty else 0.0

        if not np.isfinite(mu):
            mu = 0.0

        mu = float(np.clip(mu, -0.05, 0.05))

    return [float(last_close * np.exp(mu * i)) for i in range(1, horizon_days + 1)]


def _forecast_tabular(
    df: pd.DataFrame,
    model: str,
    horizon_days: int,
) -> tuple[list[float], list[str]]:
    clean = clean_ohlcv(df)

    if len(clean) < 80:
        return (
            _drift_forecast(clean, horizon_days),
            ["Not enough history for a fitted tabular model; used drift fallback."],
        )

    x_last, _ = latest_feature_vector(clean)
    last_close = float(clean["Close"].iloc[-1])

    forecasts: list[float] = []
    warnings: list[str] = []

    for h in range(1, horizon_days + 1):
        ds = make_supervised_dataset(clean, h)

        if len(ds.y) < 60:
            fallback_tail = _drift_forecast(clean, horizon_days - h + 1)
            forecasts.extend(fallback_tail)
            warnings.append(
                f"Horizon {h}: not enough fitted samples; "
                "used drift fallback for the remaining points."
            )
            break

        try:
            pred = float(_fit_predict_tabular(model, ds.X, ds.y, x_last)[0])
        except ModelTrainingError:
            raise
        except Exception as exc:
            fallback_tail = _drift_forecast(clean, horizon_days - h + 1)
            forecasts.extend(fallback_tail)
            warnings.append(
                f"{_model_display_name(model)} failed for horizon {h}: {exc}. "
                "Used drift fallback for the remaining points."
            )
            break

        pred = clipped_log_return(pred, h)
        forecasts.append(float(last_close * np.exp(pred)))

    return forecasts[:horizon_days], warnings


def _train_lstm_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 70,
) -> object:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise ModelTrainingError(
            "PyTorch is required for the LSTM model. Install torch or choose PDT/XGBoost."
        ) from exc

    torch.manual_seed(42)
    torch.set_num_threads(1)

    class _LSTMRegressor(nn.Module):
        def __init__(self, input_dim: int, horizon: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=32,
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(p=0.10)
            self.head = nn.Sequential(
                nn.Linear(32, 24),
                nn.ReLU(),
                nn.Linear(24, horizon),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            last = self.dropout(out[:, -1, :])
            return self.head(last)

    device = torch.device("cpu")

    model = _LSTMRegressor(
        input_dim=X_train.shape[-1],
        horizon=Y_train.shape[1],
    ).to(device)

    x_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(Y_train, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_state = None
    best_loss = float("inf")
    patience = 10
    stale = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())

        if loss_value + 1e-8 < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def _lstm_predict(model: object, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        x_tensor = torch.as_tensor(X, dtype=torch.float32)
        pred = model(x_tensor)

    return pred.detach().cpu().numpy()


def _forecast_lstm(
    df: pd.DataFrame,
    horizon_days: int,
) -> tuple[list[float], list[str]]:
    clean = clean_ohlcv(df)
    sequence_length = 30

    if len(clean) < sequence_length + horizon_days + 50:
        return (
            _drift_forecast(clean, horizon_days),
            ["Not enough history for LSTM; used drift fallback."],
        )

    X, Y, _, _, _ = make_lstm_dataset(
        clean,
        horizon_days=horizon_days,
        sequence_length=sequence_length,
    )

    if len(X) < 60:
        return (
            _drift_forecast(clean, horizon_days),
            ["Not enough LSTM sequences; used drift fallback."],
        )

    X_last, _ = latest_lstm_sequence(clean, sequence_length=sequence_length)

    X_scaled, X_last_scaled, _, _ = _standardize_sequences(X, X_last)
    assert X_last_scaled is not None

    model = _train_lstm_model(X_scaled, Y, epochs=70)
    pred = _lstm_predict(model, X_last_scaled).reshape(-1)

    last_close = float(clean["Close"].iloc[-1])

    values: list[float] = []
    for i, raw_pred in enumerate(pred[:horizon_days], start=1):
        log_ret = clipped_log_return(float(raw_pred), i)
        values.append(float(last_close * np.exp(log_ret)))

    return values, []


def evaluate_model(
    df: pd.DataFrame,
    model: str,
    horizon_days: int = 1,
    test_fraction: float = 0.2,
) -> MetricResult:
    clean = clean_ohlcv(df)
    horizon_days = int(max(1, horizon_days))

    if model == "naive":
        return _evaluate_naive(clean, horizon_days, test_fraction)

    if model == "lstm":
        return _evaluate_lstm(clean, horizon_days, test_fraction)

    if model in {"pdt", "xgb"}:
        return _evaluate_tabular(clean, model, horizon_days, test_fraction)

    raise ValueError(f"Unknown model: {model}")


def _metric_from_predictions(
    model: str,
    horizon_days: int,
    train_size: int,
    base_close: np.ndarray,
    target_close: np.ndarray,
    true_log_return: np.ndarray,
    pred_log_return: np.ndarray,
) -> MetricResult:
    pred_log_return = np.asarray(
        [clipped_log_return(float(x), horizon_days) for x in pred_log_return],
        dtype=float,
    )

    pred_price = base_close * np.exp(pred_log_return)
    naive_price = base_close

    errors = pred_price - target_close
    naive_errors = naive_price - target_close

    denom = np.where(np.abs(target_close) > 1e-12, np.abs(target_close), np.nan)
    mape = np.nanmean(np.abs(errors) / denom) * 100.0

    directional_accuracy = (
        np.mean(np.sign(pred_log_return) == np.sign(true_log_return)) * 100.0
    )

    return MetricResult(
        model=model,
        horizon_days=horizon_days,
        train_size=int(train_size),
        test_size=int(len(target_close)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        mape=float(0.0 if not np.isfinite(mape) else mape),
        directional_accuracy=float(directional_accuracy),
        naive_rmse=float(np.sqrt(np.mean(naive_errors**2))),
        naive_mae=float(np.mean(np.abs(naive_errors))),
    )


def _chronological_split(n: int, test_fraction: float) -> int:
    test_size = max(20, int(round(n * test_fraction)))
    split = max(1, n - test_size)

    if split < 40 and n >= 80:
        split = n - 40

    return split


def _evaluate_tabular(
    df: pd.DataFrame,
    model: str,
    horizon_days: int,
    test_fraction: float,
) -> MetricResult:
    ds = make_supervised_dataset(df, horizon_days)

    if len(ds.y) < 80:
        return _evaluate_naive(df, horizon_days, test_fraction, model_name=model)

    split = _chronological_split(len(ds.y), test_fraction)

    if split <= 0 or split >= len(ds.y):
        return _evaluate_naive(df, horizon_days, test_fraction, model_name=model)

    X_train = ds.X[:split]
    X_test = ds.X[split:]
    y_train = ds.y[:split]
    y_test = ds.y[split:]

    pred = _fit_predict_tabular(model, X_train, y_train, X_test)

    return _metric_from_predictions(
        model=model,
        horizon_days=horizon_days,
        train_size=split,
        base_close=ds.base_close[split:],
        target_close=ds.target_close[split:],
        true_log_return=y_test,
        pred_log_return=pred,
    )


def _evaluate_naive(
    df: pd.DataFrame,
    horizon_days: int,
    test_fraction: float,
    model_name: str = "naive",
) -> MetricResult:
    ds = make_supervised_dataset(df, horizon_days)

    if len(ds.y) < 5:
        raise ModelTrainingError("Not enough data to evaluate even the naive baseline")

    split = _chronological_split(len(ds.y), test_fraction)
    split = min(max(1, split), len(ds.y) - 1)

    y_test = ds.y[split:]
    pred = np.zeros_like(y_test)

    return _metric_from_predictions(
        model=model_name,
        horizon_days=horizon_days,
        train_size=split,
        base_close=ds.base_close[split:],
        target_close=ds.target_close[split:],
        true_log_return=y_test,
        pred_log_return=pred,
    )


def _evaluate_lstm(
    df: pd.DataFrame,
    horizon_days: int,
    test_fraction: float,
) -> MetricResult:
    sequence_length = 30

    X, Y, base, target_prices, _ = make_lstm_dataset(
        df,
        horizon_days,
        sequence_length=sequence_length,
    )

    if len(X) < 90:
        return _evaluate_naive(df, horizon_days, test_fraction, model_name="lstm")

    split = _chronological_split(len(X), test_fraction)
    split = min(max(40, split), len(X) - 1)

    X_train = X[:split]
    X_test = X[split:]
    Y_train = Y[:split]
    Y_test = Y[split:]

    X_train_s, X_test_s, _, _ = _standardize_sequences(X_train, X_test)
    assert X_test_s is not None

    model = _train_lstm_model(X_train_s, Y_train, epochs=60)
    pred_all = _lstm_predict(model, X_test_s)

    col = horizon_days - 1

    return _metric_from_predictions(
        model="lstm",
        horizon_days=horizon_days,
        train_size=split,
        base_close=base[split:],
        target_close=target_prices[split:, col],
        true_log_return=Y_test[:, col],
        pred_log_return=pred_all[:, col],
    )


def forecast_from_history(
    df: pd.DataFrame,
    model: str,
    horizon_days: int,
) -> ForecastResult:
    model = model.lower().strip()

    if model not in {"pdt", "lstm", "xgb", "naive"}:
        raise ValueError(f"Unknown model: {model}")

    horizon_days = int(max(1, horizon_days))

    clean = clean_ohlcv(df)

    if clean.empty or "Close" not in clean.columns:
        raise ModelTrainingError("No valid close-price history")

    warnings: list[str] = []

    if model == "naive":
        values = _drift_forecast(clean, horizon_days, zero_return=True)

    elif model in {"pdt", "xgb"}:
        values, w = _forecast_tabular(clean, model, horizon_days)
        warnings.extend(w)

    elif model == "lstm":
        try:
            values, w = _forecast_lstm(clean, horizon_days)
            warnings.extend(w)
        except ModelTrainingError as exc:
            values = _drift_forecast(clean, horizon_days)
            warnings.append(str(exc) + " Used drift fallback.")

    else:
        raise ValueError(f"Unknown model: {model}")

    metrics: MetricResult | None = None

    try:
        metrics = evaluate_model(
            clean,
            model,
            horizon_days=min(horizon_days, 30),
            test_fraction=0.2,
        )
    except Exception as exc:
        warnings.append(f"Backtest metrics are unavailable: {exc}")

    return ForecastResult(
        values=[float(x) for x in values],
        metrics=metrics,
        warnings=warnings,
        model_display_name=_model_display_name(model),
    )


def evaluate_many(
    df: pd.DataFrame,
    models: list[str],
    horizon_days: int = 1,
) -> list[MetricResult]:
    results: list[MetricResult] = []

    for model in models:
        try:
            results.append(
                evaluate_model(
                    df,
                    model,
                    horizon_days=horizon_days,
                    test_fraction=0.2,
                )
            )
        except ModelTrainingError:
            raise
        except Exception:
            continue

    return results
