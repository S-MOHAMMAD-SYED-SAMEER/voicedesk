"""The safety boundary: a database whose name does not end in `_evals`.

The evaluation runner creates, migrates and truncates whatever it is pointed
at. Every test here is about the one rule that keeps that from being a
catastrophe, and the rule is checked on the resolved name so that a derived
URL is policed exactly as a configured one is.
"""

import pytest
from sqlalchemy.engine import make_url

from app.config import Settings
from app.evals.database import (
    EVAL_SUFFIX,
    EVAL_TABLES,
    EvalDatabaseError,
    require_eval_database,
    resolve_eval_database_url,
)

REAL = "postgresql+psycopg://voicedesk:voicedesk@localhost:5432/voicedesk"
EVALS = "postgresql+psycopg://voicedesk:voicedesk@localhost:5432/voicedesk_evals"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- the boundary ----------------------------------------------------------


def test_an_evals_database_is_allowed() -> None:
    assert require_eval_database(EVALS).database == "voicedesk_evals"


def test_the_development_database_is_refused() -> None:
    with pytest.raises(EvalDatabaseError, match="Refusing to use database"):
        require_eval_database(REAL)


def test_the_refusal_explains_why() -> None:
    with pytest.raises(EvalDatabaseError, match="creates, migrates and truncates"):
        require_eval_database(REAL)


def test_a_name_that_merely_contains_the_suffix_is_refused() -> None:
    """It has to end with it. `evals_backup` is somebody's backup."""
    with pytest.raises(EvalDatabaseError):
        require_eval_database(REAL.replace("voicedesk", "voicedesk_evals_backup"))


def test_a_url_with_no_database_is_refused() -> None:
    with pytest.raises(EvalDatabaseError):
        require_eval_database("postgresql+psycopg://user:pass@localhost:5432/")


def test_the_suffix_is_what_it_says_it_is() -> None:
    assert EVAL_SUFFIX == "_evals"


# --- resolution ------------------------------------------------------------


def test_a_configured_evaluation_database_wins() -> None:
    settings = _settings(database_url=REAL, eval_database_url=EVALS)

    assert make_url(resolve_eval_database_url(settings)).database == "voicedesk_evals"


def test_a_configured_database_is_policed_like_any_other() -> None:
    settings = _settings(database_url=EVALS, eval_database_url=REAL)

    with pytest.raises(EvalDatabaseError, match="Refusing"):
        resolve_eval_database_url(settings)


def test_one_is_derived_when_none_is_configured() -> None:
    settings = _settings(database_url=REAL)

    assert make_url(resolve_eval_database_url(settings)).database == "voicedesk_evals"


def test_derivation_keeps_the_rest_of_the_url() -> None:
    settings = _settings(database_url=REAL)
    url = make_url(resolve_eval_database_url(settings))

    assert (url.host, url.port, url.username) == ("localhost", 5432, "voicedesk")
    assert url.drivername == "postgresql+psycopg"


def test_a_database_already_named_for_evaluation_is_not_doubled() -> None:
    settings = _settings(database_url=EVALS)

    assert make_url(resolve_eval_database_url(settings)).database == "voicedesk_evals"


def test_whitespace_around_a_configured_url_is_ignored() -> None:
    settings = _settings(database_url=REAL, eval_database_url=f"  {EVALS}  ")

    assert make_url(resolve_eval_database_url(settings)).database == "voicedesk_evals"


def test_a_url_naming_no_database_cannot_be_derived_from() -> None:
    settings = _settings(database_url="postgresql+psycopg://user@localhost:5432/")

    with pytest.raises(EvalDatabaseError, match="names no database"):
        resolve_eval_database_url(settings)


def test_something_that_is_not_a_url_fails_clearly() -> None:
    settings = _settings(database_url=REAL, eval_database_url="not a url at all")

    with pytest.raises(EvalDatabaseError, match="not a database URL"):
        resolve_eval_database_url(settings)


def test_the_default_settings_derive_an_evaluation_database() -> None:
    """A fresh clone can run the suite without configuring anything."""
    assert resolve_eval_database_url(_settings()).endswith(EVAL_SUFFIX)


# --- what gets emptied -----------------------------------------------------


def test_every_table_the_system_writes_is_truncated() -> None:
    """A table left out would leak one scenario's rows into the next."""
    from app.db.base import Base

    assert set(EVAL_TABLES) == set(Base.metadata.tables)


def test_truncation_refuses_a_database_it_may_not_touch(monkeypatch) -> None:
    """The check runs again immediately before anything is deleted."""
    from sqlalchemy import create_engine

    from app.evals.database import truncate

    engine = create_engine(REAL)
    try:
        with pytest.raises(EvalDatabaseError, match="Refusing"):
            truncate(engine)
    finally:
        engine.dispose()


def test_migration_refuses_a_database_it_may_not_touch() -> None:
    from app.evals.database import migrate

    with pytest.raises(EvalDatabaseError, match="Refusing"):
        migrate(REAL)


def test_creation_refuses_a_database_it_may_not_touch() -> None:
    from app.evals.database import ensure_database

    with pytest.raises(EvalDatabaseError, match="Refusing"):
        ensure_database(REAL)
