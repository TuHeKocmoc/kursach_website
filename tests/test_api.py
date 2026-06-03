import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def synthetic_ohlcv(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 20000.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.025, size=n)))
    open_ = close * (1 + rng.normal(0, 0.004, size=n))
    high = np.maximum(open_, close) * (1 + rng.random(n) * 0.012)
    low = np.minimum(open_, close) * (1 - rng.random(n) * 0.012)
    volume = rng.integers(10000, 50000, size=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def test_health_history_and_predict(monkeypatch) -> None:
    df = synthetic_ohlcv()

    import app.api.endpoints.market as market_endpoint
    import app.api.endpoints.predict as predict_endpoint

    monkeypatch.setattr(market_endpoint, "download_history", lambda *args, **kwargs: df)
    monkeypatch.setattr(market_endpoint, "download_last_price", lambda *args, **kwargs: df.tail(1))
    monkeypatch.setattr(predict_endpoint, "download_history", lambda *args, **kwargs: df)

    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}

    history = client.get("/api/market/history?symbol=BTC-USD&period=1y&interval=1d")
    assert history.status_code == 200
    payload = history.json()
    assert payload["symbol"] == "BTC-USD"
    assert len(payload["candles"]) == len(df)
    assert "tenkan_sen" in payload["candles"][-1]

    prediction = client.post(
        "/api/predict",
        json={"symbol": "BTC-USD", "model": "pdt", "horizon_days": 3, "interval": "1d"},
    )
    assert prediction.status_code == 200
    pred_payload = prediction.json()
    assert pred_payload["model"] == "pdt"
    assert len(pred_payload["forecast"]) == 3
    assert pred_payload["metrics"] is not None