"""The credential that proves a media socket belongs to a call we answered.

The webhook is signed; the WebSocket upgrade after it is not, and the carrier
offers nothing to sign it with. So the webhook mints a short-lived token
bound to the `CallSid`, puts it in the `<Stream>` URL, and the socket checks
it.

Two things these tests hold in place. The signature check is unchanged —
nothing here relaxes it to make a token easier to test. And the token is a
secret: it is never logged, on any path.
"""

import logging
import pathlib
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app.config import Settings, get_settings
from app.telephony import stream_token
from app.telephony.security import SIGNATURE_HEADER, expected_signature
from app.telephony.twiml import connect_stream, stream_url

from .conftest import (
    TWILIO_AUTH_TOKEN,
    TWILIO_CALL_SID,
    TWILIO_STREAM_SID,
    FakeModel,
    FakeSTT,
    FakeTTS,
    say,
    start_frame,
)

OTHER_SID = "CA00000000000000000000000000000002"
BASE = "https://voicedesk.example.com"
VOICE = "/telephony/voice"
STREAM = "/telephony/stream"
PUBLIC_URL = f"{BASE}{VOICE}"
FORM = {
    "CallSid": TWILIO_CALL_SID,
    "AccountSid": "AC1",
    "From": "+447700900123",
    "To": "+441234567890",
}


# --- minting and checking --------------------------------------------------


def test_a_freshly_minted_token_is_valid() -> None:
    token = stream_token.mint(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 300)

    assert stream_token.is_valid(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, token)


def test_a_token_carries_its_version_and_expiry() -> None:
    """The shape is deliberate, so a later version cannot be read as this one."""
    token = stream_token.mint(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 300, now=1000.0)
    version, expires, signature = token.split(stream_token.SEPARATOR)

    assert version == stream_token.VERSION
    assert expires == "1300"
    assert signature


def test_a_token_is_bound_to_one_call() -> None:
    """The property that makes this worth doing at all."""
    token = stream_token.mint(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 300)

    assert stream_token.is_valid(TWILIO_AUTH_TOKEN, OTHER_SID, token) is False


def test_a_token_expires() -> None:
    token = stream_token.mint(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 60, now=1000.0)

    assert stream_token.is_valid(
        TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, token, now=1059.0
    )
    assert (
        stream_token.is_valid(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, token, now=1061.0)
        is False
    )


def test_the_expiry_cannot_be_moved_without_the_key() -> None:
    """It is inside the signed payload, not merely printed beside it."""
    token = stream_token.mint(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 60, now=1000.0)
    _version, _expires, signature = token.split(stream_token.SEPARATOR)
    forged = f"{stream_token.VERSION}.{int(time.time()) + 99999}.{signature}"

    assert stream_token.is_valid(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, forged) is False


def test_a_token_minted_under_another_key_is_refused() -> None:
    token = stream_token.mint("somebody-elses-token", TWILIO_CALL_SID, 300)

    assert stream_token.is_valid(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, token) is False


@pytest.mark.parametrize(
    "token",
    [None, "", "nonsense", "v1.notanumber.sig", "v1.1000", "v0.1000.sig", "a.b.c.d"],
)
def test_everything_malformed_fails_closed(token) -> None:
    assert stream_token.is_valid(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, token) is False


def test_no_auth_token_means_no_valid_token() -> None:
    assert stream_token.is_valid("", TWILIO_CALL_SID, "anything") is False


def test_no_call_sid_means_no_valid_token() -> None:
    token = stream_token.mint(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 300)

    assert stream_token.is_valid(TWILIO_AUTH_TOKEN, "", token) is False


def test_minting_without_a_key_refuses_rather_than_signing_with_nothing() -> None:
    with pytest.raises(stream_token.StreamTokenError, match="auth token"):
        stream_token.mint("", TWILIO_CALL_SID, 300)


def test_minting_without_a_call_refuses() -> None:
    with pytest.raises(stream_token.StreamTokenError):
        stream_token.mint(TWILIO_AUTH_TOKEN, "", 300)


def test_two_calls_get_different_tokens() -> None:
    first = stream_token.mint(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 300, now=1000.0)
    second = stream_token.mint(TWILIO_AUTH_TOKEN, OTHER_SID, 300, now=1000.0)

    assert first != second


def test_the_comparison_is_constant_time() -> None:
    """Read from the source: a timing-safe comparison is the whole point."""
    source = pathlib.Path(stream_token.__file__).read_text()

    assert "compare_digest" in source
    assert "== signature" not in source


def test_the_token_module_logs_nothing() -> None:
    """A credential that reached a log file would be a credential at rest."""
    source = pathlib.Path(stream_token.__file__).read_text()

    assert "logger" not in source
    assert "logging" not in source


# --- the URL it travels in -------------------------------------------------


def test_a_stream_url_carries_the_token() -> None:
    assert stream_url(BASE, token="v1.123.abc").endswith("?token=v1.123.abc")


def test_a_stream_url_without_a_token_is_what_it_always_was() -> None:
    """Milestone 6's contract, unchanged when no token is minted."""
    assert stream_url(BASE) == "wss://voicedesk.example.com/telephony/stream"


def test_the_twiml_escapes_the_token_into_the_attribute() -> None:
    document = connect_stream(BASE, token="v1.123.ab+c/d")

    assert "ab%2Bc%2Fd" in document
    assert document.startswith("<?xml")


# --- end to end: webhook to socket -----------------------------------------


@pytest.fixture
def signed(migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch):
    """A client with signature checking on, as production requires."""
    from app.main import create_app

    def build(**overrides):
        fields = {
            "telephony_enabled": True,
            "twilio_auth_token": TWILIO_AUTH_TOKEN,
            "public_base_url": BASE,
            "validate_twilio_signature": True,
            "business_timezone": "UTC",
        }
        fields.update(overrides)
        settings = Settings(_env_file=None, **fields)
        get_settings.cache_clear()
        monkeypatch.setattr("app.config.get_settings", lambda: settings)
        monkeypatch.setattr("app.telephony.webhook.get_settings", lambda: settings)
        monkeypatch.setattr("app.telephony.stream.get_settings", lambda: settings)
        monkeypatch.setattr(
            "app.telephony.stream.build_model",
            lambda settings=None: FakeModel(say("Certainly.")),
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_stt", lambda settings=None: FakeSTT()
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_tts", lambda settings=None: FakeTTS()
        )
        return TestClient(create_app())

    return build


def _answer(client: TestClient) -> str:
    """Place a signed call and return the token out of the TwiML."""
    import re

    signature = expected_signature(TWILIO_AUTH_TOKEN, PUBLIC_URL, FORM)
    response = client.post(VOICE, data=FORM, headers={SIGNATURE_HEADER: signature})
    assert response.status_code == 200, response.text
    found = re.search(r"token=([^&\"]+)", response.text)
    assert found, response.text
    return found.group(1)


def test_a_signed_webhook_hands_back_a_token(signed) -> None:
    token = _answer(signed())

    assert stream_token.is_valid(TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, token)


def test_the_socket_accepts_that_token(signed, twilio_call, open_weekdays) -> None:
    client = signed()
    token = _answer(client)

    with client.websocket_connect(f"{STREAM}?token={token}") as socket:
        socket.send_text(start_frame())

        assert socket.receive_json()["event"] in {"media", "mark"}


def test_the_socket_refuses_a_missing_token(signed, twilio_call, open_weekdays) -> None:
    """An attacker who learns the address still has nothing."""
    import starlette.websockets

    client = signed()
    _answer(client)

    with pytest.raises(starlette.websockets.WebSocketDisconnect) as disconnected:
        with client.websocket_connect(STREAM) as socket:
            socket.send_text(start_frame())
            socket.receive_json()

    assert disconnected.value.code == 1008


def test_the_socket_refuses_a_token_minted_for_another_call(
    signed, twilio_call, open_weekdays
) -> None:
    """Guessing a `CallSid` is no longer enough."""
    import starlette.websockets

    client = signed()
    other = stream_token.mint(TWILIO_AUTH_TOKEN, OTHER_SID, 300)

    with pytest.raises(starlette.websockets.WebSocketDisconnect):
        with client.websocket_connect(f"{STREAM}?token={other}") as socket:
            socket.send_text(start_frame())
            socket.receive_json()


def test_the_socket_refuses_an_expired_token(
    signed, twilio_call, open_weekdays
) -> None:
    import starlette.websockets

    client = signed()
    stale = stream_token.mint(
        TWILIO_AUTH_TOKEN, TWILIO_CALL_SID, 1, now=time.time() - 600
    )

    with pytest.raises(starlette.websockets.WebSocketDisconnect):
        with client.websocket_connect(f"{STREAM}?token={stale}") as socket:
            socket.send_text(start_frame())
            socket.receive_json()


class _Capture(logging.Handler):
    """Records from one logger, kept whole.

    Not `caplog`: Alembic's `fileConfig` disables every logger that already
    exists when a migration runs, and this suite migrates before it gets
    here. A capture that silently recorded nothing would make a
    "no secret in the log" assertion pass by being empty, which is the one
    way it must not pass.
    """

    def __init__(self, name: str) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []
        self._logger = logging.getLogger(name)
        self._disabled = self._logger.disabled
        self._level = self._logger.level

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    def __enter__(self) -> "_Capture":
        self._logger.disabled = False
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self)
        return self

    def __exit__(self, *_exception) -> None:
        self._logger.removeHandler(self)
        self._logger.disabled = self._disabled
        self._logger.setLevel(self._level)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def test_a_refused_token_is_not_written_to_the_log(
    signed, twilio_call, open_weekdays
) -> None:
    """The refusal says a token was wrong. It never says which."""
    import starlette.websockets

    client = signed()
    other = stream_token.mint(TWILIO_AUTH_TOKEN, OTHER_SID, 300)

    with _Capture("app.telephony.stream") as captured:
        with pytest.raises(starlette.websockets.WebSocketDisconnect):
            with client.websocket_connect(f"{STREAM}?token={other}") as socket:
                socket.send_text(start_frame())
                socket.receive_json()

    # The positive half: something was written, so the negative half means
    # something.
    assert any("stream token" in line for line in captured.lines), captured.lines
    assert other not in captured.text
    assert TWILIO_AUTH_TOKEN not in captured.text


def test_the_minted_token_is_not_written_to_the_log(signed) -> None:
    with _Capture("app.telephony.webhook") as captured:
        token = _answer(signed())

    assert token not in captured.text


def test_the_call_identifier_is_still_in_the_log(
    signed, twilio_call, open_weekdays
) -> None:
    """What a log line may carry: the stream this refusal was about."""
    import starlette.websockets

    client = signed()
    other = stream_token.mint(TWILIO_AUTH_TOKEN, OTHER_SID, 300)

    with _Capture("app.telephony.stream") as captured:
        with pytest.raises(starlette.websockets.WebSocketDisconnect):
            with client.websocket_connect(f"{STREAM}?token={other}") as socket:
                socket.send_text(start_frame())
                socket.receive_json()

    assert TWILIO_STREAM_SID in captured.text


# --- local replay is unchanged ---------------------------------------------


def test_no_token_is_required_when_signatures_are_not_being_checked(
    signed, twilio_call, open_weekdays
) -> None:
    """Milestone 6's local-replay path, still exactly as it was.

    The switch that turns signature checking off exists for replaying
    captured requests with no auth token. Production cannot set it —
    `app/preflight.py` refuses to start — so requiring a token here would
    have bought nothing and broken every simulated test in the suite.
    """
    client = signed(validate_twilio_signature=False)

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())

        assert socket.receive_json()["event"] in {"media", "mark"}
