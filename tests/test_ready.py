"""Readiness: two checks, and the ones it deliberately does not make.

`/health` stays what it was — a liveness probe that touches nothing, so an
orchestrator never restarts a working process because PostgreSQL blinked.
`/ready` answers a different question: should this process be sent a call?

It asks the database and it asks Alembic. It asks no provider. A readiness
probe that depended on somebody else's rate limit would take every replica
out of rotation at once, for a reason that has nothing to do with whether
this service works.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app.api import ready as ready_module
from app.runtime import get_admission


# --- liveness is unchanged -------------------------------------------------


def test_health_still_touches_nothing(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert set(body) == {"status", "app", "version", "environment"}


def test_health_answers_while_the_database_is_unreachable(
    client: TestClient, monkeypatch
) -> None:
    """A dependency being down must not get a working process restarted."""
    monkeypatch.setattr(
        ready_module, "_database", lambda: ready_module.Check(ok=False, detail="down")
    )

    assert client.get("/health").status_code == 200


# --- readiness -------------------------------------------------------------


def test_ready_is_200_with_a_migrated_database(
    client: TestClient, migrated_engine: Engine
) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_reports_both_checks(
    client: TestClient, migrated_engine: Engine
) -> None:
    body = client.get("/ready").json()

    assert body["database"]["ok"] is True
    assert body["migrations"]["ok"] is True
    assert "at head" in body["migrations"]["detail"]


def test_ready_is_503_when_the_database_is_unreachable(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        ready_module,
        "_database",
        lambda: ready_module.Check(ok=False, detail="unreachable (OperationalError)"),
    )
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not ready"
    assert response.json()["database"]["ok"] is False


def test_the_migration_check_is_skipped_when_the_database_is_down(
    client: TestClient, monkeypatch
) -> None:
    """There is nothing to compare a stamped revision against."""
    monkeypatch.setattr(
        ready_module, "_database", lambda: ready_module.Check(ok=False, detail="down")
    )
    body = client.get("/ready").json()

    assert body["migrations"]["ok"] is False
    assert "not checked" in body["migrations"]["detail"]


def test_ready_is_503_when_the_database_is_behind_head(
    client: TestClient, monkeypatch
) -> None:
    """A process running ahead of its migrations writes to columns that are
    not there yet."""
    monkeypatch.setattr(
        ready_module, "_database", lambda: ready_module.Check(ok=True, detail="up")
    )
    monkeypatch.setattr(
        ready_module,
        "_migrations",
        lambda: ready_module.Check(
            ok=False, detail="database is at abc123, this build expects def456"
        ),
    )
    response = client.get("/ready")

    assert response.status_code == 503
    assert "expects" in response.json()["migrations"]["detail"]


def test_an_unmigrated_database_is_not_ready(
    client: TestClient, database_url: str, alembic_config, monkeypatch
) -> None:
    """Measured against a real empty database, not a stubbed check."""
    from alembic import command

    from app.config import get_settings
    from app.db.session import reset_engine

    monkeypatch.setenv("VOICEDESK_DATABASE_URL", database_url)
    get_settings.cache_clear()
    reset_engine()
    command.downgrade(alembic_config, "base")
    try:
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["migrations"]["ok"] is False
    finally:
        command.downgrade(alembic_config, "base")
        get_settings.cache_clear()
        reset_engine()


def test_a_draining_process_is_not_ready(
    client: TestClient, migrated_engine: Engine
) -> None:
    """Stop being sent calls before you stop answering them."""
    admission = get_admission()
    admission.start_draining()
    try:
        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["draining"] is True
    finally:
        admission.draining = False


def test_ready_reports_how_many_calls_are_in_progress(
    client: TestClient, migrated_engine: Engine
) -> None:
    assert client.get("/ready").json()["active_calls"] == 0


# --- what readiness must never do ------------------------------------------


def test_readiness_calls_no_provider() -> None:
    """Checked against the code, not the prose.

    The module docstring names all four vendors on purpose — it is the
    paragraph explaining that none of them is called. What must not exist is
    a vendor the code can reach.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(ready_module.__file__).read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith("app.providers") for name in imports)

    for node in ast.walk(tree):
        # Strip the docstring of every module, class and function.
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                body[0].value.value = ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for vendor in ("anthropic", "deepgram", "elevenlabs", "twilio"):
                assert vendor not in node.value.lower(), node.value
        if isinstance(node, ast.Name):
            for vendor in ("anthropic", "deepgram", "elevenlabs", "twilio"):
                assert vendor not in node.id.lower(), node.id


def test_readiness_never_logs_a_database_url() -> None:
    """A connection string carries a password."""
    import pathlib

    source = pathlib.Path(ready_module.__file__).read_text()

    assert "database_url" not in source


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_neither_probe_needs_authentication(client: TestClient, path: str) -> None:
    assert client.get(path).status_code in (200, 503)
