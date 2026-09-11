"""The database layer: engine, session, and that migrations really ran."""

import pytest
from sqlalchemy import Engine, inspect, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import Base
from app.db.session import get_engine, get_session, reset_engine

EXPECTED_TABLES = {
    "calls",
    "turns",
    "tool_calls",
    "appointments",
    "services",
    "business_hours",
}


def test_engine_uses_the_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "VOICEDESK_DATABASE_URL",
        "postgresql+psycopg://someone:secret@db.example:5432/other",
    )
    get_settings.cache_clear()
    reset_engine()
    try:
        url = get_engine().url
        assert url.drivername == "postgresql+psycopg"
        assert url.host == "db.example"
        assert url.database == "other"
    finally:
        get_settings.cache_clear()
        reset_engine()


def test_engine_is_reused(settings_env: None) -> None:
    assert get_engine() is get_engine()


def test_constraint_naming_convention_is_configured() -> None:
    assert Base.metadata.naming_convention["pk"] == "pk_%(table_name)s"


def test_the_connection_works(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        assert connection.execute(text("select 1")).scalar_one() == 1


def test_get_session_yields_a_usable_session(migrated_engine: Engine) -> None:
    sessions = get_session()
    db_session = next(sessions)
    try:
        assert db_session.execute(text("select 1")).scalar_one() == 1
    finally:
        sessions.close()


def test_the_migration_creates_every_table(migrated_engine: Engine) -> None:
    assert EXPECTED_TABLES <= set(inspect(migrated_engine).get_table_names())


def test_the_models_and_the_migration_agree(migrated_engine: Engine) -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_the_test_database_is_not_the_development_one(session: Session) -> None:
    database = session.execute(text("select current_database()")).scalar_one()

    assert "voicedesk" in database
    assert "docintel" not in database
