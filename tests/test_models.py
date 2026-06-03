import builtins

import numpy as np
import pandas as pd
import pytest

from app.ml.models import ModelTrainingError, _make_xgb_regressor, forecast_from_history


def test_xgb_requires_xgboost_package(monkeypatch) -> None:
    real_import = builtins.__import__

    def block_xgboost(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_xgboost)

    with pytest.raises(ModelTrainingError, match="requires the xgboost package"):
        _make_xgb_regressor()


def test_xgb_forecast_does_not_fall_back_without_xgboost(monkeypatch) -> None:
    real_import = builtins.__import__

    def block_xgboost(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    idx = pd.date_range("2023-01-01", periods=140, freq="D")
    close = 100.0 * np.exp(np.linspace(0.0, 0.2, len(idx)))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )

    monkeypatch.setattr(builtins, "__import__", block_xgboost)

    with pytest.raises(ModelTrainingError, match="requires the xgboost package"):
        forecast_from_history(df, model="xgb", horizon_days=1)
