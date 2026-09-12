"""Where the evaluation suite is allowed to run, and where it is not.

The evaluator creates databases, migrates them and truncates every table
between scenarios. That is fine against a database that exists for exactly
that purpose and catastrophic against one that does not, so this module holds
a single hard boundary:

    **A database whose name does not end in `_evals` is never touched.**

Not read, not migrated, and above all not truncated. The check is on the
resolved name, so it applies equally to a URL configured by hand and one
derived from `database_url` — there is no path through this module that
reaches another database, and `require_eval_database` is called again
immediately before the only statement that deletes anything.

Deriving rather than defaulting is deliberate. A hard-coded default would
eventually point at somebody's real database after a rename; appending to
whatever `database_url` says keeps the evaluation database beside the real one
and obviously named after it.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# The whole safety boundary, in one string.
EVAL_SUFFIX = "_evals"

# Emptied between scenarios. Ordered so that a plain `TRUNCATE` would work
# even without CASCADE; CASCADE is still used, because a table added later and
# forgotten here would otherwise fail loudly rather than silently leak rows.
EVAL_TABLES = (
    "call_costs",
    "tool_calls",
    "turns",
    "appointments",
    "calls",
    "business_hours",
    "services",
)

# Connected to only to issue `CREATE DATABASE`, which cannot run inside the
# database it creates.
MAINTENANCE_DATABASE = "postgres"


class EvalDatabaseError(RuntimeError):
    """The evaluation database is missing, unreachable, or not allowed."""


def resolve_eval_database_url(settings: Settings | None = None) -> str:
    """The database the evaluation suite may use, or a clear refusal.

    `eval_database_url` wins when it is set. Otherwise one is derived from
    `database_url` by appending `_evals` to the database name — so a
    deployment pointing at `voicedesk` evaluates against `voicedesk_evals`
    without configuring anything.
    """
    resolved = settings or get_settings()

    configured = resolved.eval_database_url.strip()
    if configured:
        url = _parse(configured, "VOICEDESK_EVAL_DATABASE_URL")
        require_eval_database(url)
        return url.render_as_string(hide_password=False)

    url = _parse(resolved.database_url, "VOICEDESK_DATABASE_URL")
    if not url.database:
        raise EvalDatabaseError(
            "VOICEDESK_DATABASE_URL names no database, so no evaluation "
            "database can be derived from it. Set "
            "VOICEDESK_EVAL_DATABASE_URL to a database whose name ends in "
            f"{EVAL_SUFFIX!r}."
        )

    name = url.database
    derived = name if name.endswith(EVAL_SUFFIX) else f"{name}{EVAL_SUFFIX}"
    url = url.set(database=derived)
    require_eval_database(url)
    return url.render_as_string(hide_password=False)


def require_eval_database(url: URL | str) -> URL:
    """Refuse anything that is not an evaluation database.

    Called on every resolution and again before the truncate. Two checks of
    one rule is cheap; deleting somebody's development data is not.
    """
    parsed = url if isinstance(url, URL) else _parse(url, "database URL")
    name = parsed.database or ""
    if not name.endswith(EVAL_SUFFIX):
        raise EvalDatabaseError(
            f"Refusing to use database {name!r}: the evaluation suite creates, "
            f"migrates and truncates whatever it is given, so it will only "
            f"touch a database whose name ends in {EVAL_SUFFIX!r}."
        )
    return parsed


def ensure_database(url: str) -> None:
    """Create the evaluation database if it is not there yet."""
    parsed = require_eval_database(url)

    engine = create_engine(url)
    try:
        with engine.connect():
            return
    except sqlalchemy.exc.OperationalError as exc:
        if not _missing_database(exc):
            raise EvalDatabaseError(
                f"Cannot reach the evaluation database {parsed.database!r}: {exc}"
            ) from exc
    finally:
        engine.dispose()

    _create(parsed)


def truncate(engine: Engine) -> None:
    """Empty every table, so one scenario cannot be seen by the next.

    `TRUNCATE` rather than `DROP`/`CREATE`: the schema is migrated once per
    run and 18 scenarios should not pay for 18 migrations.
    """
    require_eval_database(engine.url)

    statement = text(
        f"TRUNCATE {', '.join(EVAL_TABLES)} RESTART IDENTITY CASCADE"
    )
    with engine.begin() as connection:
        connection.execute(statement)


def migrate(url: str) -> None:
    """Bring the evaluation database to head, using the project's migrations.

    No migration is added for evaluation and none is needed: the suite runs
    against exactly the schema the application runs against, which is the
    point.
    """
    from alembic import command

    require_eval_database(url)
    with _alembic_config(url) as config:
        command.upgrade(config, "head")


# --- internals -------------------------------------------------------------


@contextmanager
def _alembic_config(url: str) -> Iterator[object]:
    """An Alembic config pointed at one database, and put back afterwards.

    `alembic/env.py` reads the URL from application settings on purpose, so
    that migrations and the app can never disagree. Pointing it somewhere else
    therefore means pointing the settings somewhere else for the duration —
    restored in the `finally`, cache and all.
    """
    import pathlib

    from alembic.config import Config as AlembicConfig

    root = pathlib.Path(__file__).resolve().parent.parent.parent
    config = AlembicConfig(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", url)

    previous = os.environ.get("VOICEDESK_DATABASE_URL")
    os.environ["VOICEDESK_DATABASE_URL"] = url
    get_settings.cache_clear()
    try:
        yield config
    finally:
        if previous is None:
            os.environ.pop("VOICEDESK_DATABASE_URL", None)
        else:
            os.environ["VOICEDESK_DATABASE_URL"] = previous
        get_settings.cache_clear()


def _parse(url: str, source: str) -> URL:
    try:
        return make_url(url)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same answer
        raise EvalDatabaseError(f"{source} is not a database URL: {url!r}") from exc


def _missing_database(exc: Exception) -> bool:
    return "does not exist" in str(exc)


def _create(parsed: URL) -> None:
    """`CREATE DATABASE`, from the server's maintenance database."""
    maintenance = create_engine(
        parsed.set(database=MAINTENANCE_DATABASE).render_as_string(
            hide_password=False
        ),
        isolation_level="AUTOCOMMIT",
    )
    try:
        with maintenance.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{parsed.database}"'))
        logger.info("Created the evaluation database %s.", parsed.database)
    except sqlalchemy.exc.ProgrammingError as exc:
        if "already exists" in str(exc):
            return
        raise EvalDatabaseError(
            f"Could not create the evaluation database {parsed.database!r}: {exc}"
        ) from exc
    except sqlalchemy.exc.OperationalError as exc:
        raise EvalDatabaseError(
            f"Could not reach PostgreSQL to create {parsed.database!r}: {exc}"
        ) from exc
    finally:
        maintenance.dispose()


__all__ = [
    "EVAL_SUFFIX",
    "EVAL_TABLES",
    "EvalDatabaseError",
    "ensure_database",
    "migrate",
    "require_eval_database",
    "resolve_eval_database_url",
    "truncate",
]
