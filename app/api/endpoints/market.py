from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from app.api.data_utils import normalize_download_df, ts_to_datetime
from app.api.schemas import Candle, MarketHistoryResponse, MarketLastPriceResponse
from app.ml.features import add_technical_indicators, numeric_or_none
from app.services.market_data import download_history, download_last_price

router = APIRouter(prefix="/market")


def _df_to_candles(df: pd.DataFrame) -> list[Candle]:
    candles: list[Candle] = []
    if df.empty:
        return candles
    
    enriched = add_technical_indicators(df)
    for idx, row in enriched.iterrows():
        if any(pd.isna(row.get(k)) for k in ("Open", "High", "Low", "Close")):
            continue
        t = ts_to_datetime(idx)
        if t is None:
            continue
        volume_value = row.get("Volume")
        volume = 0.0 if volume_value is None or pd.isna(volume_value) else float(volume_value)
        candles.append(
            Candle(
                time=t,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=volume,
                tenkan_sen=numeric_or_none(row.get("tenkan_sen")),
                kijun_sen=numeric_or_none(row.get("kijun_sen")),
                senkou_span_a=numeric_or_none(row.get("senkou_span_a")),
                senkou_span_b=numeric_or_none(row.get("senkou_span_b")),
                chikou_span=numeric_or_none(row.get("chikou_span")),
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
    try:
        df = download_history(symbol=symbol, period=period, interval=interval, start=start, end=end)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download market data: {e}") from e

    candles = _df_to_candles(df)
    if not candles:
        raise HTTPException(status_code=404, detail="No market data available for given parameters")


    return MarketHistoryResponse(
        symbol=symbol.upper(),
        interval=interval,
        start=candles[0].time,
        end=candles[-1].time,
        candles=candles,
    )


@router.get("/last", response_model=MarketLastPriceResponse)
def last_price(symbol: str = Query(default="BTC-USD", min_length=1)) -> MarketLastPriceResponse:
    try:
        df = download_last_price(symbol)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download last price data: {e}") from e

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
