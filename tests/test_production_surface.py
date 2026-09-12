"""What production serves, and what it will not serve at any price.

The harness is the reason this file exists. It is a page with no
authentication that opens a socket, spends model budget and writes real rows
— exactly right for development and an open door anywhere else. Production
does not serve it, and `harness_enabled` cannot make it.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

REAL_DB = "postgresql+psycopg://someone:secret@db.example:5432/voicedesk"


def _client(monkeypatch, **overrides) -> TestClient:
    fields = {
        "environment": "production",
        "database_url": REAL_DB,
        "anthropic_api_key": "sk-test-not-a-real-key",
    }
    fields.update(overrides)
    settings = Settings(_env_file=None, **fields)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.harness.get_settings", lambda: settings)
    return TestClient(create_app(settings))


# --- the harness is a development tool -------------------------------------


def test_the_harness_page_is_served_in_development(client: TestClient) -> None:
    assert client.get("/harness").status_code == 200


def test_the_harness_page_is_absent_in_production(monkeypatch) -> None:
    """404, not 403: there is nothing here to probe."""
    assert _client(monkeypatch).get("/harness").status_code == 404


def test_the_harness_socket_is_refused_in_production(monkeypatch) -> None:
    """No budget can be spent through a page that is not being served."""
    import starlette.websockets

    with _client(monkeypatch) as production:
        with pytest.raises(starlette.websockets.WebSocketDisconnect):
            with production.websocket_connect("/ws/harness") as socket:
                socket.receive_json()


def test_production_ignores_the_harness_switch(monkeypatch) -> None:
    """An operator who leaves it on must still not expose it."""
    assert _client(monkeypatch, harness_enabled=True).get("/harness").status_code == 404


def test_the_switch_still_works_outside_production(settings_env, monkeypatch) -> None:
    """A shared staging deployment can turn it off without being production."""
    settings = Settings(_env_file=None, environment="staging", harness_enabled=False)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.harness.get_settings", lambda: settings)

    assert TestClient(create_app(settings)).get("/harness").status_code == 404


def test_harness_availability_is_the_two_conditions() -> None:
    assert Settings(_env_file=None).harness_available is True
    assert Settings(_env_file=None, harness_enabled=False).harness_available is False
    assert Settings(_env_file=None, environment="production").harness_available is False


# --- the documentation surface ---------------------------------------------


def test_the_docs_are_published_in_development(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_the_docs_are_closed_in_production(monkeypatch) -> None:
    """They describe every route and schema to anybody who asks."""
    production = _client(monkeypatch)

    assert production.get("/docs").status_code == 404
    assert production.get("/redoc").status_code == 404
    assert production.get("/openapi.json").status_code == 404


def test_health_is_served_in_production(monkeypatch) -> None:
    """Closing the documentation must not close the probes."""
    assert _client(monkeypatch).get("/health").status_code == 200


# --- the error boundary ----------------------------------------------------


def test_an_unhandled_error_returns_an_opaque_body(client: TestClient) -> None:
    """A stack trace on the wire tells an attacker the shape of the system."""
    from app.main import _unhandled

    class _Request:
        method = "GET"

        class url:
            path = "/boom"

    response = _unhandled(_Request(), RuntimeError("connection to db.example failed"))
    import anyio

    body = anyio.run(lambda: _await(response))

    assert body["detail"] == "Internal Server Error"
    assert "db.example" not in str(body)
    assert "RuntimeError" not in str(body)


async def _await(coroutine):
    import json

    result = await coroutine
    return json.loads(result.body)


def test_the_opaque_body_carries_a_reference(client: TestClient) -> None:
    """So an operator holding a complaint can find the one log line."""
    import anyio

    from app.main import _unhandled

    class _Request:
        method = "POST"

        class url:
            path = "/telephony/voice"

    body = anyio.run(lambda: _await(_unhandled(_Request(), ValueError("x"))))

    assert len(body["reference"]) == 12
