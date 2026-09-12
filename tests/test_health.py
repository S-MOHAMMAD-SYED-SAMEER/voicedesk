"""The application starts and says so."""

from fastapi.testclient import TestClient


def test_the_application_builds() -> None:
    from app.main import create_app

    app = create_app()

    assert app.title == "VoiceDesk"


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "VoiceDesk"
    assert body["environment"] == "test"
    assert body["version"]


def test_health_needs_no_database(client: TestClient) -> None:
    """Liveness must not depend on Postgres being up."""
    assert client.get("/health").status_code == 200


def test_the_http_surface_is_health_ready_the_harness_and_the_voice_webhook(
    client: TestClient,
) -> None:
    """Liveness, readiness, the development page, and the call answerer.

    There is still no booking API and no dialogue API: the calendar, the tools
    and the dialogue layer are libraries, and the only ways to reach them over
    the network are the two sockets — `/ws/harness` and `/telephony/stream` —
    which do not appear here because WebSocket routes are not in OpenAPI.
    """
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {"/health", "/ready", "/harness", "/telephony/voice"}


def test_the_voice_webhook_is_absent_until_telephony_is_enabled(
    client: TestClient,
) -> None:
    """It is routed, but it answers nothing while switched off."""
    assert client.post("/telephony/voice").status_code == 404
