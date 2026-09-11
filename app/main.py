"""FastAPI application entrypoint: `uvicorn app.main:app --reload`.

`GET /health` from milestone 1, the milestone-5 development harness
(`GET /harness` and `WS /ws/harness`), and the milestone-6 telephony adapter
(`POST /telephony/voice` and `WS /telephony/stream`), which serves nothing
unless `telephony_enabled` is set.
"""

from fastapi import FastAPI

from app import __version__
from app.api import harness, health
from app.config import get_settings
from app.telephony import router as telephony_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
    )
    app.include_router(health.router)
    app.include_router(harness.router)
    app.include_router(telephony_router)
    return app


app = create_app()
