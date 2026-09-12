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
    "call_costs",
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


# --- the milestone-8 migration --------------------------------------------

MIGRATIONS = 3


def test_there_are_exactly_three_migrations() -> None:
    """One per schema change so far, and no more than one for milestone 8."""
    import pathlib

    versions = pathlib.Path(__file__).resolve().parent.parent / "alembic" / "versions"
    revisions = sorted(path.name for path in versions.glob("*.py"))

    assert len(revisions) == MIGRATIONS, revisions


def test_the_earlier_migrations_are_untouched() -> None:
    """Milestone 8 adds a revision; it does not edit history."""
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parent.parent
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "alembic/versions"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()

    for name in changed:
        assert "94468e1a40f1" not in name, name
        assert "fa76475b048b" not in name, name


def test_the_new_migration_is_safe_against_a_populated_table(
    database_url: str, alembic_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It adds a table and alters nothing, so existing rows cannot be hurt."""
    import uuid

    from alembic import command
    from sqlalchemy import create_engine

    monkeypatch.setenv("VOICEDESK_DATABASE_URL", database_url)
    get_settings.cache_clear()
    reset_engine()

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "fa76475b048b")

    engine = create_engine(database_url)
    call_id = uuid.uuid4()
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO calls (id, from_number, to_number) "
                    "VALUES (:id, '+447700900123', '+441234567890')"
                ),
                {"id": call_id},
            )

        command.upgrade(alembic_config, "head")

        with engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT count(*) FROM calls WHERE id = :id"), {"id": call_id}
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text("SELECT total_cost_usd FROM calls WHERE id = :id"),
                    {"id": call_id},
                ).scalar_one()
                is None
            )
    finally:
        engine.dispose()
        command.downgrade(alembic_config, "base")
        get_settings.cache_clear()
        reset_engine()


def test_the_new_migration_can_be_undone_and_redone(
    database_url: str, alembic_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped enum type left behind would break the next upgrade."""
    from alembic import command
    from sqlalchemy import create_engine

    monkeypatch.setenv("VOICEDESK_DATABASE_URL", database_url)
    get_settings.cache_clear()
    reset_engine()

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "fa76475b048b")

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert "call_costs" not in inspect(connection).get_table_names()
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM pg_type "
                        "WHERE typname = 'cost_component'"
                    )
                ).scalar_one()
                == 0
            )

        command.upgrade(alembic_config, "head")

        with engine.connect() as connection:
            assert "call_costs" in inspect(connection).get_table_names()
    finally:
        engine.dispose()
        command.downgrade(alembic_config, "base")
        get_settings.cache_clear()
        reset_engine()
