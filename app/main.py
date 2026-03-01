from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.ui.router import router as ui_router


def create_app() -> FastAPI:
    app = FastAPI(title="BTC Forecast")
    app.include_router(api_router)
    app.include_router(ui_router)

    static_dir = Path(__file__).resolve().parent / "ui" / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


app = create_app()