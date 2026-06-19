import pandas as pd

from app.services import market_data


def test_download_history_uses_ticker_history_when_download_is_empty(monkeypatch) -> None:
    market_data._history_cache.clear()
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    history_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=idx,
    )

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs) -> pd.DataFrame:
            return history_df

    class FakeYfinance:
        Ticker = FakeTicker

        @staticmethod
        def download(**kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(market_data, "_load_yfinance", lambda: FakeYfinance)

    df = market_data.download_history("BTC-USD", period="1y", interval="1d", use_cache=False)

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df["Close"].tolist() == [101.0, 102.0]


def test_empty_history_response_is_not_cached(monkeypatch) -> None:
    market_data._history_cache.clear()
    calls = {"count": 0}
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    good_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1000, 1100],
        },
        index=idx,
    )

    def fake_download(symbol: str, period: str | None = None, interval: str = "1d", **kwargs) -> pd.DataFrame:
        calls["count"] += 1
        if calls["count"] == 1:
            return pd.DataFrame()
        return good_df

    monkeypatch.setattr(market_data, "_download_yf", fake_download)

    first = market_data.download_history("BTC-USD", period="1y", interval="1d", use_cache=True)
    second = market_data.download_history("BTC-USD", period="1y", interval="1d", use_cache=True)

    assert first.empty
    assert second["Close"].tolist() == [101.0, 102.0]
    assert calls["count"] == 2
