from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

PRICE_FEATURE_WINDOWS = (3, 7, 14, 30)
RETURN_LAGS = (1, 2, 3, 5, 7, 14, 21, 30)
MODEL_FEATURE_COLUMNS = [
    "log_ret_1",
    "log_ret_2",
    "log_ret_3",
    "log_ret_5",
    "log_ret_7",
    "log_ret_14",
    "log_ret_21",
    "log_ret_30",
    "hl_range",
    "oc_change",
    "volume_log_change",
    "rsi_14",
    "macd_rel",
    "macd_signal_rel",
    "macd_hist_rel",
    "bb_width_20",
    "bb_position_20",
    "tenkan_rel",
    "kijun_rel",
    "senkou_a_rel",
    "senkou_b_rel",
    "close_to_ma_3",
    "close_to_ma_7",
    "close_to_ma_14",
    "close_to_ma_30",
    "rolling_vol_3",
    "rolling_vol_7",
    "rolling_vol_14",
    "rolling_vol_30",
]

DISPLAY_INDICATOR_COLUMNS = [
    "tenkan_sen",
    "kijun_sen",
    "senkou_span_a",
    "senkou_span_b",
    "chikou_span",
]


@dataclass(frozen=True)
class SupervisedDataset:
    X: np.ndarray
    y: np.ndarray
    index: pd.Index
    base_close: np.ndarray
    target_close: np.ndarray
    feature_names: list[str]


def _rename_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        for level in range(out.columns.nlevels):
            values = {str(v).lower() for v in out.columns.get_level_values(level)}
            if {"open", "high", "low", "close"}.issubset(values):
                out.columns = out.columns.get_level_values(level)
                break
        else:
            out.columns = ["_".join(str(part) for part in col if str(part)) for col in out.columns]

    mapping: dict[str, str] = {}
    for col in out.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in {"open", "high", "low", "close", "volume"}:
            mapping[col] = key.capitalize()
        elif key == "adj_close":
            mapping[col] = "Adj Close"
    return out.rename(columns=mapping)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    out = _rename_ohlcv_columns(df)
    if "Close" not in out.columns:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    for col in ("Open", "High", "Low"):
        if col not in out.columns:
            out[col] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    out = out.loc[:, list(OHLCV_COLUMNS)].copy()
    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0.0)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def _safe_divide(num: pd.Series, den: pd.Series | float, default: float = 0.0) -> pd.Series:
    result = num / den
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(default)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_ohlcv(df)
    if out.empty:
        return out

    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    open_ = out["Open"].astype(float)
    volume = out["Volume"].astype(float)

    for lag in RETURN_LAGS:
        out[f"log_ret_{lag}"] = np.log(close / close.shift(lag))

    out["hl_range"] = _safe_divide(high - low, close)
    out["oc_change"] = _safe_divide(close - open_, open_)
    out["volume_log_change"] = np.log1p(volume).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["rsi_14"] = (_rsi(close, 14) - 50.0) / 50.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_rel"] = _safe_divide(macd, close)
    out["macd_signal_rel"] = _safe_divide(macd_signal, close)
    out["macd_hist_rel"] = _safe_divide(macd - macd_signal, close)

    ma_20 = close.rolling(20, min_periods=20).mean()
    std_20 = close.rolling(20, min_periods=20).std(ddof=0)
    upper = ma_20 + 2.0 * std_20
    lower = ma_20 - 2.0 * std_20
    out["bb_width_20"] = _safe_divide(upper - lower, ma_20)
    out["bb_position_20"] = _safe_divide(close - lower, upper - lower, default=0.5)

    for window in PRICE_FEATURE_WINDOWS:
        ma = close.rolling(window, min_periods=window).mean()
        out[f"close_to_ma_{window}"] = _safe_divide(close - ma, ma)
        out[f"rolling_vol_{window}"] = out["log_ret_1"].rolling(window, min_periods=window).std(ddof=0)

    tenkan = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2.0
    kijun = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2.0).shift(26)
    chikou = close.shift(-26)

    out["tenkan_sen"] = tenkan
    out["kijun_sen"] = kijun
    out["senkou_span_a"] = span_a
    out["senkou_span_b"] = span_b
    out["chikou_span"] = chikou

    out["tenkan_rel"] = _safe_divide(tenkan - close, close)
    out["kijun_rel"] = _safe_divide(kijun - close, close)
    out["senkou_a_rel"] = _safe_divide(span_a - close, close)
    out["senkou_b_rel"] = _safe_divide(span_b - close, close)

    return out.replace([np.inf, -np.inf], np.nan)


def model_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_technical_indicators(df)
    if enriched.empty:
        return pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
    features = enriched.reindex(columns=MODEL_FEATURE_COLUMNS).copy()
    return features.ffill().fillna(0.0)


def latest_feature_vector(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    features = model_feature_frame(df)
    if features.empty:
        raise ValueError("Not enough market data to build features")
    return features.iloc[[-1]].to_numpy(dtype=float), list(features.columns)


def make_supervised_dataset(df: pd.DataFrame, horizon_days: int) -> SupervisedDataset:
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    enriched = add_technical_indicators(df)
    if enriched.empty:
        return SupervisedDataset(
            X=np.empty((0, len(MODEL_FEATURE_COLUMNS))),
            y=np.empty((0,)),
            index=pd.Index([]),
            base_close=np.empty((0,)),
            target_close=np.empty((0,)),
            feature_names=list(MODEL_FEATURE_COLUMNS),
        )

    features = enriched.reindex(columns=MODEL_FEATURE_COLUMNS).ffill().fillna(0.0)
    close = enriched["Close"].astype(float)
    target = np.log(close.shift(-horizon_days) / close)
    target_close = close.shift(-horizon_days)
    data = features.copy()
    data["__target__"] = target
    data["__base_close__"] = close
    data["__target_close__"] = target_close
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["__target__", "__base_close__", "__target_close__"])

    if data.empty:
        return SupervisedDataset(
            X=np.empty((0, len(MODEL_FEATURE_COLUMNS))),
            y=np.empty((0,)),
            index=pd.Index([]),
            base_close=np.empty((0,)),
            target_close=np.empty((0,)),
            feature_names=list(MODEL_FEATURE_COLUMNS),
        )

    return SupervisedDataset(
        X=data[MODEL_FEATURE_COLUMNS].to_numpy(dtype=float),
        y=data["__target__"].to_numpy(dtype=float),
        index=data.index,
        base_close=data["__base_close__"].to_numpy(dtype=float),
        target_close=data["__target_close__"].to_numpy(dtype=float),
        feature_names=list(MODEL_FEATURE_COLUMNS),
    )


def make_lstm_dataset(
    df: pd.DataFrame,
    horizon_days: int,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    enriched = add_technical_indicators(df)
    if enriched.empty:
        return (
            np.empty((0, sequence_length, len(MODEL_FEATURE_COLUMNS))),
            np.empty((0, horizon_days)),
            np.empty((0,)),
            np.empty((0, horizon_days)),
            list(MODEL_FEATURE_COLUMNS),
        )

    features = enriched.reindex(columns=MODEL_FEATURE_COLUMNS).ffill().fillna(0.0).to_numpy(dtype=float)
    close = enriched["Close"].to_numpy(dtype=float)

    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    base: list[float] = []
    target_prices: list[np.ndarray] = []
    last_end = len(close) - horizon_days - 1
    for end_pos in range(sequence_length - 1, last_end + 1):
        c0 = close[end_pos]
        future = close[end_pos + 1 : end_pos + horizon_days + 1]
        if not np.isfinite(c0) or c0 <= 0 or len(future) != horizon_days or not np.all(np.isfinite(future)):
            continue
        X.append(features[end_pos - sequence_length + 1 : end_pos + 1])
        Y.append(np.log(future / c0))
        base.append(float(c0))
        target_prices.append(future.astype(float))

    if not X:
        return (
            np.empty((0, sequence_length, len(MODEL_FEATURE_COLUMNS))),
            np.empty((0, horizon_days)),
            np.empty((0,)),
            np.empty((0, horizon_days)),
            list(MODEL_FEATURE_COLUMNS),
        )

    return (
        np.stack(X).astype(float),
        np.stack(Y).astype(float),
        np.asarray(base, dtype=float),
        np.stack(target_prices).astype(float),
        list(MODEL_FEATURE_COLUMNS),
    )


def latest_lstm_sequence(df: pd.DataFrame, sequence_length: int = 30) -> tuple[np.ndarray, list[str]]:
    features = model_feature_frame(df)
    if len(features) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} rows to build an LSTM sequence")
    return features.tail(sequence_length).to_numpy(dtype=float)[None, :, :], list(features.columns)


def clipped_log_return(prediction: float, horizon_days: int) -> float:
    if not np.isfinite(prediction):
        return 0.0
    bound = min(0.75, 0.20 * float(np.sqrt(max(1, horizon_days))))
    return float(np.clip(prediction, -bound, bound))


def numeric_or_none(value: object) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None