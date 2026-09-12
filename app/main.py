"""FastAPI application entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

`GET /health` (liveness) and `GET /ready` (readiness), the development harness
at `GET /harness` and `WS /ws/harness`, and the telephony adapter at
`POST /telephony/voice` and `WS /telephony/stream`. Telephony serves nothing
unless `telephony_enabled` is set; the harness serves nothing in production,
whatever is set.

Four things happen around the application, and each exists because of a
specific way a deployment can go wrong.

* **Preflight.** In production, missing configuration stops the process at
  boot rather than surfacing as a 503 to the first caller. See
  `app/preflight.py`. Outside production nothing is required, because the
  offline providers, the test suite and the evaluation suite all have to run
  with no credential at all.
* **Draining.** On shutdown the process stops accepting calls and gives the
  ones in progress a bounded grace period. It cannot do better than bounded:
  a model request already running on a worker thread cannot be cancelled or
  killed, so the wait is a wait and not a guarantee.
* **A closed documentation surface.** `/docs`, `/redoc` and `/openapi.json`
  describe every route and schema to anybody who asks. Useful in development,
  not something to publish.
* **An opaque error boundary.** An unhandled exception returns a reference
  and nothing else; the traceback goes to the log, where the operator is.

Migrations are **not** run from here. `alembic upgrade head` is a deployment
step, run once, before the new version starts — not something each replica
races the others to do.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import harness, health, ready
from app.config import Settings, get_settings
from app.logging import configure as configure_logging
from app.preflight import check as preflight
from app.runtime import get_admission
from app.telephony import router as telephony_router

logger = logging.getLogger(__name__)

# How often the drain loop looks to see whether the last call has ended.
DRAIN_POLL_SECONDS = 0.25


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    if resolved.cost_tracking_enabled:
        # Raises `PricingError` on a price that is not a number, is negative,
        # or is so large it can only be a misplaced decimal point.
        from app.cost import PriceBook

        PriceBook.from_settings(resolved)

    preflight(resolved)

    app = FastAPI(
        title=resolved.app_name,
        version=__version__,
        debug=resolved.debug,
        lifespan=_lifespan,
        # Published in development, closed in production.
        docs_url=None if resolved.is_production else "/docs",
        redoc_url=None if resolved.is_production else "/redoc",
        openapi_url=None if resolved.is_production else "/openapi.json",
    )
    app.include_router(health.router)
    app.include_router(ready.router)
    app.include_router(harness.router)
    app.include_router(telephony_router)
    app.add_exception_handler(Exception, _unhandled)
    return app


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on the way up; drain on the way down."""
    del app
    settings = get_settings()
    configure_logging(settings)
    logger.info(
        "VoiceDesk %s starting: environment=%s telephony=%s realtime=%s "
        "harness=%s stt=%s tts=%s max_calls=%d",
        __version__,
        settings.environment,
        settings.telephony_enabled,
        settings.realtime_enabled,
        settings.harness_available,
        settings.stt_provider,
        settings.tts_provider,
        settings.max_concurrent_calls,
    )

    try:
        yield
    finally:
        await _drain(settings)


async def _drain(settings: Settings) -> None:
    """Stop taking calls, wait a bounded while for the rest, let go.

    Honest about its limit: this waits for calls to end. It cannot end them.
    A turn whose model request is already running on a worker thread will
    finish that request whatever happens here — `anyio.to_thread.run_sync`
    abandons a thread, it does not kill one — so a call still running when
    the grace period expires is left, logged, and the process stops anyway.
    """
    from app.db.session import reset_engine

    admission = get_admission()
    admission.start_draining()

    if admission.active:
        logger.info(
            "Draining: %d call(s) in progress, waiting up to %.0fs.",
            admission.active,
            settings.shutdown_grace_seconds,
        )
        with anyio.move_on_after(settings.shutdown_grace_seconds):
            while admission.active:
                await anyio.sleep(DRAIN_POLL_SECONDS)

    if admission.active:
        logger.warning(
            "Stopping with %d call(s) still in progress: the grace period "
            "expired and a running model request cannot be cancelled.",
            admission.active,
        )
    else:
        logger.info("Drained cleanly.")

    reset_engine()


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Everything the client is told about an unexpected failure.

    A reference and a sentence. No exception type, no message, no traceback,
    no SQL: a stack trace on the wire tells an attacker the shape of the
    system, and a database error tells them its schema. The reference is
    echoed into the log beside the whole traceback, so an operator holding a
    complaint can find the one line that matters.
    """
    reference = uuid.uuid4().hex[:12]
    logger.exception(
        "Unhandled error on %s %s [%s]",
        request.method,
        request.url.path,
        reference,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "reference": reference},
    )


app = create_app()
