"""FastAPI application entrypoint: `uvicorn app.main:app --reload`.

`GET /health` from milestone 1, and the milestone-5 development harness:
`GET /harness` for the page and `WS /ws/harness` for the call itself.
Telephony arrives with the milestone that builds it.
"""

from fastapi import FastAPI

from app import __version__
from app.api import harness, health
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
    )
    app.include_router(health.router)
    app.include_router(harness.router)
    return app


app = create_app()
