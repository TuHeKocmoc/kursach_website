import numpy as np
import pandas as pd

from app.ml.features import add_technical_indicators, make_supervised_dataset, model_feature_frame


def synthetic_ohlcv(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, size=n)))
    open_ = close * (1 + rng.normal(0, 0.004, size=n))
    high = np.maximum(open_, close) * (1 + rng.random(n) * 0.01)
    low = np.minimum(open_, close) * (1 - rng.random(n) * 0.01)
    volume = rng.integers(1000, 5000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_features_and_supervised_dataset_are_built() -> None:
    df = synthetic_ohlcv()
    enriched = add_technical_indicators(df)

    assert "tenkan_sen" in enriched.columns
    assert "kijun_sen" in enriched.columns
    assert "chikou_span" in enriched.columns

    features = model_feature_frame(df)
    assert not features.empty
    assert features.isna().sum().sum() == 0

    ds = make_supervised_dataset(df, horizon_days=7)
    assert ds.X.shape[0] == ds.y.shape[0]
    assert ds.X.shape[1] == len(ds.feature_names)
    assert ds.X.shape[0] > 100