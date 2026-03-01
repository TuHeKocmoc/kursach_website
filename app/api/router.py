from fastapi import APIRouter

from app.api.endpoints.health import router as health_router
from app.api.endpoints.market import router as market_router
from app.api.endpoints.predict import router as predict_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router, tags=["health"])
api_router.include_router(market_router, tags=["market"])
api_router.include_router(predict_router, tags=["predict"])