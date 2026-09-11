"""FastAPI application entrypoint: `uvicorn app.main:app --reload`.

Milestone 1 serves one endpoint. Telephony, the browser harness and the
dialogue API arrive with the milestones that build them.
"""

from fastapi import FastAPI

from app import __version__
from app.api import health
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
    )
    app.include_router(health.router)
    return app


app = create_app()
