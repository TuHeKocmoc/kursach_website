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
    tenkan_sen: Optional[float] = None
    kijun_sen: Optional[float] = None
    senkou_span_a: Optional[float] = None
    senkou_span_b: Optional[float] = None
    chikou_span: Optional[float] = None


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
    model: Literal["pdt", "lstm", "xgb", "naive"] = "pdt"
    horizon_days: int = Field(default=1, ge=1, le=365)
    interval: Literal["1d"] = "1d"


class PredictionPoint(BaseModel):
    time: datetime
    value: float


class BacktestMetrics(BaseModel):
    model: str
    horizon_days: int
    train_size: int
    test_size: int
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    naive_rmse: float
    naive_mae: float


class PredictResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    model_display_name: str = Field(..., min_length=1)
    horizon_days: int = Field(..., ge=1)
    generated_at: datetime
    last_time: datetime
    last_price: float
    forecast: List[PredictionPoint]
    metrics: Optional[BacktestMetrics] = None
    warnings: List[str] = Field(default_factory=list)


class EvaluateResponse(BaseModel):
    symbol: str = Field(..., min_length=1)
    horizon_days: int = Field(..., ge=1)
    generated_at: datetime
    metrics: List[BacktestMetrics]


class ModelInfo(BaseModel):
    name: Literal["pdt", "lstm", "xgb", "naive"]
    label: str
    description: str


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
