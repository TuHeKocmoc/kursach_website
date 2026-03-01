from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException

from app.api.data_utils import as_utc, normalize_download_df, ts_to_datetime
from app.api.schemas import PredictRequest, PredictResponse, PredictionPoint

router = APIRouter()


def _mean_daily_return(close: pd.Series, window: int = 30) -> float:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 3:
        return 0.0
    r = s.pct_change().dropna()
    if r.empty:
        return 0.0
    r = r.tail(window)
    m = float(r.mean())
    if not pd.isna(m) and abs(m) < 1.0:
        return m
    return 0.0


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    df_raw = yf.download(
        tickers=req.symbol,
        period="2y",
        interval=req.interval,
        progress=False,
        auto_adjust=False,
    )
    df = normalize_download_df(df_raw, req.symbol)
    if df.empty or "Close" not in df.columns:
        raise HTTPException(status_code=404, detail="No market data available for prediction")

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise HTTPException(status_code=404, detail="No market data available for prediction")

    last_idx = df.index[-1]
    last_time = ts_to_datetime(last_idx)
    if last_time is None:
        raise HTTPException(status_code=404, detail="No market data available for prediction")

    last_close = float(df.iloc[-1]["Close"])
    mu = _mean_daily_return(df["Close"], window=30)

    forecast: list[PredictionPoint] = []
    for i in range(1, req.horizon_days + 1):
        t = last_time + timedelta(days=i)
        v = last_close * ((1.0 + mu) ** i)
        forecast.append(PredictionPoint(time=as_utc(t), value=float(v)))

    return PredictResponse(
        symbol=req.symbol,
        model=req.model,
        horizon_days=req.horizon_days,
        generated_at=as_utc(datetime.now(timezone.utc)),
        forecast=forecast,
    )
