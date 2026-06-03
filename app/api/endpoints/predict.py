from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from app.api.data_utils import as_utc, normalize_download_df, ts_to_datetime
from app.api.schemas import PredictRequest, PredictResponse, PredictionPoint, ModelsResponse, ModelInfo, BacktestMetrics, EvaluateResponse
from app.ml.features import clean_ohlcv
from app.ml.models import forecast_from_history, ModelTrainingError, evaluate_many
from app.services.market_data import download_history

router = APIRouter()

@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(
        models=[
            ModelInfo(
                name="pdt",
                label="PDT",
                description="Permutation Decision Tree"
            ),
            ModelInfo(
                name="lstm",
                label="LSTM",
                description="Long Short-Term Memory neural network"
            ),
            ModelInfo(
                name="xgb",
                label="XGBoost",
                description="Extreme Gradient Boosting"
            ),
            ModelInfo(
                name="naive",
                label="Naive",
                description="Last-price baseline used for comparison"
            ),
        ]
    )


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
    try:
        df = download_history(symbol=req.symbol, period="5y", interval=req.interval, use_cache=False)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download market data: {e}") from e
    
    clean = clean_ohlcv(df).dropna(subset=["Close"])
    if clean.empty:
        raise HTTPException(status_code=404, detail="No market data available for prediction")

    last_idx = df.index[-1]
    last_time = ts_to_datetime(last_idx)
    if last_time is None:
        raise HTTPException(status_code=404, detail="No market data available for prediction")

    last_close = float(df.iloc[-1]["Close"])

    try:
        result = forecast_from_history(
            clean,
            model=req.model,
            horizon_days=req.horizon_days,
        )
    except (ValueError, ModelTrainingError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


    forecast = [
        PredictionPoint(
            time=as_utc(last_time + timedelta(days=i)),
            value=float(value),
        )
        for i, value in enumerate(result.values, start=1)
    ]
    metrics = BacktestMetrics(**result.metrics.to_dict()) if result.metrics is not None else None

    return PredictResponse(
        symbol=req.symbol.upper(),
        model=req.model,
        model_display_name=result.model_display_name,
        horizon_days=req.horizon_days,
        generated_at=as_utc(datetime.now(timezone.utc)),
        last_time=last_time,
        last_price=last_close,
        forecast=forecast,
        metrics=metrics,
        warnings=result.warnings,
    )

@router.get("/evaluate", response_model=EvaluateResponse)
def evaluate(
    symbol: str = Query(default="BTC-USD", min_length=1),
    horizon_days: int = Query(default=1, ge=1, le=365),
) -> EvaluateResponse:
    try:
        df = download_history(symbol=symbol, period="5y", interval="1d", use_cache=False)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download market data: {e}") from e
    
    clean = clean_ohlcv(df)
    if clean.empty:
        raise HTTPException(status_code=404, detail="No market data available for evaluation")

    metrics = [
        BacktestMetrics(**m.to_dict())
        for m in evaluate_many(
            clean,
            ["naive", "pdt", "lstm", "xgb"],
            horizon_days,
        )
    ]
    if not metrics:
        raise HTTPException(status_code=404, detail="No market data available for evaluation")
    return EvaluateResponse(
        symbol=symbol.upper(),
        horizon_days=horizon_days,
        generated_at=as_utc(datetime.now(timezone.utc)),
        metrics=metrics,
)