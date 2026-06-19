from datetime import datetime
from typing import Any

import pandas as pd

from app.ml.features import clean_ohlcv

_history_cache: dict[tuple[str, str, str], pd.DataFrame] = {}


def _load_yfinance() -> Any:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError(
            "yfinance is not installed. Install project requirements before requesting live market data."
        ) from exc
    return yf


def _download_yf(
    symbol: str,
    period: str | None = None,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    yf = _load_yfinance()
    download_kwargs: dict[str, Any] = {
        "tickers": symbol,
        "interval": interval,
        "progress": False,
        "auto_adjust": False,
        "threads": False,
    }
    history_kwargs: dict[str, Any] = {
        "interval": interval,
        "auto_adjust": False,
    }
    if start is not None or end is not None:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
        history_kwargs["start"] = start
        history_kwargs["end"] = end
    else:
        download_kwargs["period"] = period or "1y"
        history_kwargs["period"] = period or "1y"

    raw = yf.download(**download_kwargs)
    clean = clean_ohlcv(raw)
    if not clean.empty:
        return clean

    raw = yf.Ticker(symbol).history(**history_kwargs)
    return clean_ohlcv(raw)


def download_history_cached(symbol: str, period: str, interval: str) -> pd.DataFrame:
    key = (symbol, period, interval)
    cached = _history_cache.get(key)
    if cached is not None:
        return cached.copy()

    df = _download_yf(symbol=symbol, period=period, interval=interval)
    if not df.empty:
        if len(_history_cache) >= 128:
            _history_cache.pop(next(iter(_history_cache)))
        _history_cache[key] = df.copy()
    return df


def download_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    interval = interval.strip()
    period = period.strip()
    if start is None and end is None and use_cache:
        return download_history_cached(symbol, period, interval).copy()
    return _download_yf(symbol=symbol, period=period, interval=interval, start=start, end=end)


def download_last_price(symbol: str) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    attempts = [("1d", "1m"), ("5d", "1h"), ("5d", "1d")]
    last_error: Exception | None = None
    for period, interval in attempts:
        try:
            df = _download_yf(symbol=symbol, period=period, interval=interval)
        except Exception as exc:
            last_error = exc
            continue
        if not df.empty and "Close" in df.columns:
            df = df.dropna(subset=["Close"])
            if not df.empty:
                return df
    if last_error is not None:
        raise RuntimeError(str(last_error)) from last_error
    return pd.DataFrame()
