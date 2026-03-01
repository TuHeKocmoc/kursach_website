from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from app.api.data_utils import normalize_download_df, ts_to_datetime
from app.api.schemas import Candle, MarketHistoryResponse, MarketLastPriceResponse

router = APIRouter(prefix="/market")


def _df_to_candles(df: pd.DataFrame) -> list[Candle]:
    candles: list[Candle] = []
    if df.empty:
        return candles
    for idx, row in df.iterrows():
        if any(pd.isna(row.get(k)) for k in ("Open", "High", "Low", "Close")):
            continue
        t = ts_to_datetime(idx)
        if t is None:
            continue
        v = row.get("Volume")
        volume = 0.0 if v is None or pd.isna(v) else float(v)
        candles.append(
            Candle(
                time=t,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=volume,
            )
        )
    return candles


@router.get("/history", response_model=MarketHistoryResponse)
def history(
    symbol: str = Query(default="BTC-USD", min_length=1),
    interval: str = Query(default="1d", min_length=1),
    period: str = Query(default="1y", min_length=1),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> MarketHistoryResponse:
    if start is not None or end is not None:
        df_raw = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
    else:
        df_raw = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

    df = normalize_download_df(df_raw, symbol)
    candles = _df_to_candles(df)
    if not candles:
        raise HTTPException(status_code=404, detail="No market data available for given parameters")

    return MarketHistoryResponse(
        symbol=symbol,
        interval=interval,
        start=candles[0].time,
        end=candles[-1].time,
        candles=candles,
    )


@router.get("/last", response_model=MarketLastPriceResponse)
def last_price(symbol: str = Query(default="BTC-USD", min_length=1)) -> MarketLastPriceResponse:
    attempts: list[tuple[str, str]] = [
        ("1d", "1m"),
        ("5d", "1h"),
        ("5d", "1d"),
    ]

    df = pd.DataFrame()
    for period, interval in attempts:
        df_raw = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        df_try = normalize_download_df(df_raw, symbol)
        if not df_try.empty:
            df = df_try
            break

    if df.empty:
        raise HTTPException(status_code=404, detail="No last price available")

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise HTTPException(status_code=404, detail="No last price available")

    last_idx = df.index[-1]
    t = ts_to_datetime(last_idx)
    if t is None:
        raise HTTPException(status_code=404, detail="No last price available")

    price = float(df.iloc[-1]["Close"])
    return MarketLastPriceResponse(symbol=symbol, time=t, price=price)
