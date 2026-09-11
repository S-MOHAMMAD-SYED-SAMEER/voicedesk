import os
from collections.abc import Iterator

import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import reset_engine
from app.main import create_app

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tests never touch the development database. Override with
# VOICEDESK_TEST_DATABASE_URL to point at a different server.
TEST_DATABASE_URL = os.environ.get(
    "VOICEDESK_TEST_DATABASE_URL",
    "postgresql+psycopg://voicedesk:voicedesk@localhost:5432/voicedesk_test",
)


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give a test a clean settings/engine cache and restore it afterwards."""
    monkeypatch.setenv("VOICEDESK_ENVIRONMENT", "test")
    get_settings.cache_clear()
    reset_engine()
    try:
        yield
    finally:
        get_settings.cache_clear()
        reset_engine()


@pytest.fixture
def client(settings_env: None) -> Iterator[TestClient]:
    yield TestClient(create_app())


@pytest.fixture(scope="session")
def database_url() -> str:
    """The test database URL, skipping the test if no server is reachable."""
    engine = create_engine(TEST_DATABASE_URL)
    try:
        with engine.connect():
            pass
    except sqlalchemy.exc.OperationalError as exc:
        pytest.skip(f"no PostgreSQL at {TEST_DATABASE_URL}: {exc}")
    finally:
        engine.dispose()
    return TEST_DATABASE_URL


@pytest.fixture
def alembic_config(database_url: str) -> AlembicConfig:
    config = AlembicConfig(os.path.join(PROJECT_ROOT, "alembic.ini"))
    config.set_main_option("script_location", os.path.join(PROJECT_ROOT, "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture
def migrated_engine(
    database_url: str,
    alembic_config: AlembicConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Engine]:
    """A database migrated to head, torn back down to empty afterwards.

    Running the real migration rather than `Base.metadata.create_all` is the
    point: the exclusion constraint that prevents double booking exists only
    in the migration, so a schema built any other way would not have it.
    """
    monkeypatch.setenv("VOICEDESK_DATABASE_URL", database_url)
    get_settings.cache_clear()
    reset_engine()

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()
        command.downgrade(alembic_config, "base")
        get_settings.cache_clear()
        reset_engine()


@pytest.fixture
def session(migrated_engine: Engine) -> Iterator[Session]:
    with Session(migrated_engine) as db_session:
        yield db_session
