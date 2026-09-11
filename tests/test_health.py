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


def test_the_http_surface_is_health_and_the_harness(client: TestClient) -> None:
    """`/health` from milestone 1, and the milestone-5 development page.

    There is still no booking API, no dialogue API and no telephony webhook:
    the calendar, the tools and the dialogue layer are libraries, and the only
    way to reach them over the network is the harness socket.
    """
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {"/health", "/harness"}
