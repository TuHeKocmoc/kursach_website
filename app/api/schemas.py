from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class Candle(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketHistoryResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    interval: str = Field(..., min_length=1)
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    candles: List[Candle]


class MarketLastPriceResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    time: datetime
    price: float


class PredictRequest(BaseModel):
    symbol: str = Field(default="BTC-USD", min_length=1)
    model: Literal["pdt", "lstm", "xgb"] = "pdt"
    horizon_days: int = Field(default=1, ge=1, le=365)
    interval: Literal["1d"] = "1d"


class PredictionPoint(BaseModel):
    time: datetime
    value: float


class PredictResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    horizon_days: int = Field(..., ge=1)
    generated_at: datetime
    forecast: List[PredictionPoint]