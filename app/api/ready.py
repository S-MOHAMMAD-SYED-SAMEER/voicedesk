"""Readiness: is this process fit to be sent a call right now?

Deliberately different from `/health`, which stays what it was — a liveness
probe that touches nothing and answers as long as the process is running. A
liveness probe that failed on a dependency would have an orchestrator
restarting a perfectly healthy process because something else was down.

Readiness asks two questions and no others:

1. **Can it reach the database?** One `SELECT 1`. Every call writes rows from
   its first turn, so a process that cannot reach PostgreSQL cannot take a
   call, however alive it is.
2. **Is the schema the one this code expects?** The stamped Alembic revision
   compared with the head this checkout contains. A process running ahead of
   its migrations writes to columns that are not there yet; one running
   behind is usually a deployment half-done.

**No provider is called.** Not Anthropic, not Deepgram, not ElevenLabs, not
the carrier. A readiness probe that depended on somebody else's rate limit
would take this service out of rotation for a reason that has nothing to do
with whether it works, and would do it to every replica at once.

Draining counts as not ready: a process on its way out should stop being sent
new calls before it stops answering the ones it has.
"""

import logging
import pathlib
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.config import Settings, get_settings
from app.db.session import get_engine
from app.runtime import get_admission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class Check(BaseModel):
    ok: bool
    detail: str = ""


class ReadyResponse(BaseModel):
    status: Literal["ready", "not ready"]
    database: Check
    migrations: Check
    draining: bool
    active_calls: int


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
)
def ready(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadyResponse:
    """200 when this process can take a call, 503 when it cannot."""
    del settings  # Read through the checks themselves.

    database = _database()
    migrations = _migrations() if database.ok else Check(
        ok=False, detail="not checked: the database is unreachable"
    )
    admission = get_admission()

    ok = database.ok and migrations.ok and not admission.draining
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadyResponse(
        status="ready" if ok else "not ready",
        database=database,
        migrations=migrations,
        draining=admission.draining,
        active_calls=admission.active,
    )


# --- the two checks --------------------------------------------------------


def _database() -> Check:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        # The message, never the URL: a connection string carries a password.
        logger.warning("Readiness: the database is unreachable (%s).", type(exc).__name__)
        return Check(ok=False, detail=f"unreachable ({type(exc).__name__})")
    return Check(ok=True, detail="reachable")


def _migrations() -> Check:
    """The stamped revision against the head this checkout contains."""
    try:
        from alembic.config import Config as AlembicConfig
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        config = AlembicConfig(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "alembic"))
        head = ScriptDirectory.from_config(config).get_current_head()

        with get_engine().connect() as connection:
            current = MigrationContext.configure(connection).get_current_revision()
    except Exception as exc:  # noqa: BLE001 - any failure is the same answer
        logger.warning("Readiness: cannot read the migration state (%s).", type(exc).__name__)
        return Check(ok=False, detail=f"unknown ({type(exc).__name__})")

    if current == head:
        return Check(ok=True, detail=f"at head ({head})")
    return Check(
        ok=False,
        detail=f"database is at {current or 'nothing'}, this build expects {head}",
    )


__all__ = ["Check", "ReadyResponse", "ready", "router"]
