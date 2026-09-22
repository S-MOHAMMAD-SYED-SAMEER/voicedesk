"""Bounds on everything a stranger controls, and on how long a call may last.

Milestone 6 refused an unsigned webhook and a `CallSid` with no row, which
answers *who*. None of it answered *how much*: a body of any size, a control
frame of any size, calls without number, and a socket that could stay open
for ever holding a database connection.

Each bound below is a number in `Settings` with a reason written beside it.
None of them is a rate limit — there is no rate limiting here, and the README
says so.
"""

import json
import time

import pytest
import starlette.websockets
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Call, Turn
from app.runtime import get_admission, reset_admission
from app.telephony.security import SIGNATURE_HEADER, expected_signature

from .conftest import (
    TWILIO_AUTH_TOKEN,
    TWILIO_CALL_SID,
    FakeModel,
    FakeSTT,
    FakeTTS,
    media_text_frame,
    mulaw_frame,
    say,
    start_frame,
    twilio_frame,
    use_tools,
    wav_bytes,
)

VOICE = "/telephony/voice"
STREAM = "/telephony/stream"
BASE = "https://voicedesk.example.com"
PUBLIC_URL = f"{BASE}{VOICE}"
TRY_AGAIN_LATER = 1013
FORM = {
    "CallSid": TWILIO_CALL_SID,
    "AccountSid": "AC1",
    "From": "+447700900123",
    "To": "+441234567890",
}
LOUD = mulaw_frame(9000)
QUIET = mulaw_frame(0)


# --- the webhook body ------------------------------------------------------


@pytest.fixture
def webhook(migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch):
    from app.main import create_app

    def build(**overrides):
        fields = {
            "telephony_enabled": True,
            "twilio_auth_token": TWILIO_AUTH_TOKEN,
            "public_base_url": BASE,
            "validate_twilio_signature": True,
        }
        fields.update(overrides)
        settings = Settings(_env_file=None, **fields)
        get_settings.cache_clear()
        monkeypatch.setattr("app.config.get_settings", lambda: settings)
        monkeypatch.setattr("app.telephony.webhook.get_settings", lambda: settings)
        return TestClient(create_app())

    return build


def _signed(client: TestClient, form: dict):
    signature = expected_signature(TWILIO_AUTH_TOKEN, PUBLIC_URL, form)
    return client.post(VOICE, data=form, headers={SIGNATURE_HEADER: signature})


def test_a_normal_webhook_is_still_answered(webhook) -> None:
    assert _signed(webhook(), FORM).status_code == 200


def test_an_oversized_body_is_refused(webhook) -> None:
    """A voice webhook is a few hundred bytes. This is not one."""
    response = _signed(webhook(), {**FORM, "Filler": "x" * 70_000})

    assert response.status_code == 413


def test_the_limit_is_configurable(webhook) -> None:
    assert _signed(webhook(max_webhook_bytes=64), FORM).status_code == 413


def test_an_oversized_body_creates_no_call(
    session: Session, webhook, monkeypatch
) -> None:
    """Refused before it is parsed, so nothing downstream sees it."""
    _signed(webhook(), {**FORM, "Filler": "x" * 70_000})

    assert session.execute(select(Call)).first() is None


def test_an_oversized_body_is_refused_before_the_signature_is_checked(
    webhook,
) -> None:
    """Not a weakening: an unsigned body is still refused, with 403. This is
    about not doing HMAC work over a megabyte somebody sent to make us."""
    response = webhook().post(VOICE, data={**FORM, "Filler": "x" * 70_000})

    assert response.status_code == 413


def test_an_unsigned_normal_request_is_still_refused(webhook) -> None:
    """The milestone-6 contract, asserted here so the size check cannot be
    mistaken for a replacement for it."""
    assert webhook().post(VOICE, data=FORM).status_code == 403


# --- the stream frame ------------------------------------------------------


@pytest.fixture
def telephony(
    migrated_engine: Engine, telephony_settings, monkeypatch: pytest.MonkeyPatch
):
    from app.main import create_app

    def build(*responses, **overrides):
        settings = (
            telephony_settings.model_copy(update=overrides)
            if overrides
            else telephony_settings
        )
        monkeypatch.setattr("app.telephony.stream.get_settings", lambda: settings)
        monkeypatch.setattr(
            "app.telephony.stream.build_model",
            lambda settings=None: FakeModel(*(responses or (say("Certainly."),))),
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_stt", lambda settings=None: FakeSTT()
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_tts", lambda settings=None: FakeTTS()
        )
        return TestClient(create_app())

    return build


def _drain_audio(socket) -> dict:
    while True:
        message = socket.receive_json()
        if message["event"] != "media":
            return message


def test_an_oversized_frame_is_ignored_and_the_call_goes_on(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    """Ignored, not fatal: one bad frame must not end a caller's call."""
    client = telephony()

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)

        socket.send_text(json.dumps({"event": "media", "filler": "x" * 200_000}))

        # The caller speaks, and then stops. If the oversized frame had
        # broken the call this would never be answered.
        for _ in range(3):
            socket.send_text(media_text_frame(LOUD))
        for _ in range(50):
            socket.send_text(media_text_frame(QUIET))

        assert _drain_audio(socket)["event"] == "mark"


def test_an_oversized_frame_writes_no_transcript(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    client = telephony()

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        socket.send_text(json.dumps({"event": "media", "filler": "x" * 200_000}))

    assert session.execute(select(Turn)).first() is None


def test_the_frame_limit_is_configurable(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    client = telephony(max_stream_frame_bytes=32)

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())

        # Every frame including the start frame is over the limit, so the
        # call never binds and no audio is ever sent.
        socket.send_text(twilio_frame("stop", streamSid="MZ1"))


# --- how long a call may last ----------------------------------------------


def test_a_call_that_has_run_too_long_ends(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    """A socket nobody closed must not hold a database connection for ever.

    Closed deliberately, with a normal-closure code, so the carrier can tell
    an ended call from a dropped one.
    """
    client = telephony(max_call_seconds=1)

    with pytest.raises(starlette.websockets.WebSocketDisconnect) as disconnected:
        with client.websocket_connect(STREAM) as socket:
            socket.send_text(start_frame())
            _drain_audio(socket)
            time.sleep(1.1)
            socket.send_text(media_text_frame(LOUD))
            socket.receive_json()

    assert disconnected.value.code == 1000


def test_the_expiry_is_measured_from_when_the_socket_opened() -> None:
    from app.telephony.stream import StreamState

    state = StreamState(settings=Settings(_env_file=None, max_call_seconds=3600))

    assert state.expired is False

    state.opened_at -= 3601

    assert state.expired is True


def test_an_hour_is_the_default() -> None:
    assert Settings(_env_file=None).max_call_seconds == 3600


# --- how many calls at once ------------------------------------------------


def test_the_second_concurrent_call_is_refused(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    """A busy signal beats a caller waiting on a connection that is not
    coming: each call holds a database connection for its whole length."""
    client = telephony()
    admission = get_admission()
    admission.limit = 1
    try:
        with client.websocket_connect(STREAM) as first:
            first.send_text(start_frame())
            _drain_audio(first)

            with pytest.raises(
                starlette.websockets.WebSocketDisconnect
            ) as disconnected:
                with client.websocket_connect(STREAM) as second:
                    second.receive_json()

            assert disconnected.value.code == TRY_AGAIN_LATER
    finally:
        reset_admission()


def test_a_refusal_is_counted(telephony, twilio_call, open_weekdays) -> None:
    client = telephony()
    admission = get_admission()
    admission.limit = 0
    try:
        with pytest.raises(starlette.websockets.WebSocketDisconnect):
            with client.websocket_connect(STREAM) as socket:
                socket.receive_json()

        assert admission.refused == 1
    finally:
        reset_admission()


def test_the_place_comes_back_when_the_call_ends(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    client = telephony()
    admission = get_admission()
    try:
        with client.websocket_connect(STREAM) as socket:
            socket.send_text(start_frame())
            _drain_audio(socket)

        for _ in range(50):
            if admission.active == 0:
                break
            time.sleep(0.02)

        assert admission.active == 0
    finally:
        reset_admission()


def test_a_draining_process_refuses_a_new_call(
    telephony, twilio_call, open_weekdays
) -> None:
    """Stop being sent calls before you stop answering them."""
    client = telephony()
    admission = get_admission()
    admission.start_draining()
    try:
        with pytest.raises(starlette.websockets.WebSocketDisconnect) as disconnected:
            with client.websocket_connect(STREAM) as socket:
                socket.receive_json()

        assert disconnected.value.code == TRY_AGAIN_LATER
    finally:
        reset_admission()


# --- hanging up -----------------------------------------------------------


def test_hanging_up_mid_call_records_the_end(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The caller closing the socket is the ordinary way a call ends."""
    client = telephony()

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)

    session.expire_all()
    assert session.get(Call, twilio_call.id).ended_at is not None


def test_hanging_up_gives_the_place_back(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    client = telephony()
    admission = get_admission()
    try:
        with client.websocket_connect(STREAM) as socket:
            socket.send_text(start_frame())
            _drain_audio(socket)
            socket.send_text(twilio_frame("stop", streamSid="MZ1"))

        for _ in range(50):
            if admission.active == 0:
                break
            time.sleep(0.02)

        assert admission.active == 0
    finally:
        reset_admission()


# --- draining the process --------------------------------------------------


@pytest.mark.anyio
async def test_draining_returns_promptly_when_nothing_is_in_progress() -> None:
    import anyio

    from app.main import _drain

    reset_admission()
    try:
        with anyio.fail_after(2):
            await _drain(Settings(_env_file=None, shutdown_grace_seconds=30))

        assert get_admission().draining is True
    finally:
        reset_admission()


@pytest.mark.anyio
async def test_draining_gives_up_rather_than_waiting_for_ever() -> None:
    """It waits for calls to end. It cannot end them, and does not pretend
    to: a model request already on a worker thread finishes whatever
    happens here."""
    import anyio

    from app.main import _drain

    reset_admission()
    admission = get_admission()
    try:
        with admission.admit():
            with anyio.fail_after(3):
                await _drain(Settings(_env_file=None, shutdown_grace_seconds=0.2))

            assert admission.active == 1
    finally:
        reset_admission()


@pytest.mark.anyio
async def test_draining_says_so_when_it_gives_up() -> None:
    """Honest in the log, not only in the docstring.

    Captured with a handler of this test's own rather than through `caplog`:
    Alembic's `fileConfig` disables every logger that already exists when a
    migration runs, and this suite migrates long before it gets here. The
    application undoes that at startup — `configure` re-enables them — but
    `_drain` is being called here without a lifespan around it.
    """
    import logging

    from app.main import _drain

    written: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            written.append(record.getMessage())

    logger = logging.getLogger("app.main")
    handler = _Capture()
    logger.addHandler(handler)
    was_disabled, logger.disabled = logger.disabled, False

    reset_admission()
    admission = get_admission()
    try:
        with admission.admit():
            await _drain(Settings(_env_file=None, shutdown_grace_seconds=0.1))

        assert any("cannot be cancelled" in line for line in written), written
    finally:
        logger.removeHandler(handler)
        logger.disabled = was_disabled
        reset_admission()


# --- the browser harness's own bounds ---------------------------------------
#
# Telephony's `max_call_seconds` and the absence of any turn cap are both
# unchanged above. These two settings are read nowhere but
# `app/api/harness.py`, and exist for the same reason as the telephony bounds
# above: a demo left running should not hold a connection, or spend model
# budget, for ever. Neither is a rate limit or authentication — see the
# README.


@pytest.fixture
def harness(
    migrated_engine: Engine, calendar_settings, monkeypatch: pytest.MonkeyPatch
):
    """A `TestClient` whose harness uses scripted providers and, optionally,
    overridden settings — the same shape as the `telephony` fixture above."""
    from app.main import create_app

    def build(*responses, **overrides):
        settings = calendar_settings.model_copy(update=overrides)
        scripted = FakeModel(*responses)
        monkeypatch.setattr("app.api.harness.get_settings", lambda: settings)
        monkeypatch.setattr(
            "app.api.harness.build_model", lambda settings=None: scripted
        )
        monkeypatch.setattr(
            "app.api.harness.build_stt", lambda settings=None: FakeSTT()
        )
        monkeypatch.setattr(
            "app.api.harness.build_tts", lambda settings=None: FakeTTS()
        )
        return TestClient(create_app()), scripted

    return build


def test_a_harness_session_that_has_run_too_long_ends(
    harness, open_weekdays, haircut
) -> None:
    """The harness equivalent of `test_a_call_that_has_run_too_long_ends`:
    closed deliberately, with a normal-closure code, once a demo session has
    been open longer than the configured bound."""
    client, _ = harness(say("should never be used"), harness_max_session_seconds=1)

    with pytest.raises(starlette.websockets.WebSocketDisconnect) as disconnected:
        with client.websocket_connect("/ws/harness") as socket:
            socket.receive_json()  # ready
            socket.receive_bytes()  # greeting
            time.sleep(1.1)
            socket.send_bytes(wav_bytes(300))
            socket.receive_json()

    assert disconnected.value.code == 1000


def test_the_harness_expiry_is_measured_from_when_the_session_opened() -> None:
    from app.api.harness import _HarnessBudget

    budget = _HarnessBudget()
    settings = Settings(_env_file=None, harness_max_session_seconds=3600)

    assert budget.expired(settings) is False

    budget.opened_at -= 3601

    assert budget.expired(settings) is True


def test_five_minutes_is_the_harness_session_default() -> None:
    assert Settings(_env_file=None).harness_max_session_seconds == 300


def test_admission_is_released_after_the_harness_session_expires(
    harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("should never be used"), harness_max_session_seconds=1)
    admission = get_admission()
    try:
        with pytest.raises(starlette.websockets.WebSocketDisconnect):
            with client.websocket_connect("/ws/harness") as socket:
                socket.receive_json()
                socket.receive_bytes()
                time.sleep(1.1)
                socket.send_bytes(wav_bytes(300))
                socket.receive_json()

        for _ in range(50):
            if admission.active == 0:
                break
            time.sleep(0.02)

        assert admission.active == 0
    finally:
        reset_admission()


def test_the_harness_session_records_its_end_when_the_duration_limit_fires(
    session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("should never be used"), harness_max_session_seconds=1)

    with pytest.raises(starlette.websockets.WebSocketDisconnect):
        with client.websocket_connect("/ws/harness") as socket:
            socket.receive_json()
            socket.receive_bytes()
            time.sleep(1.1)
            socket.send_bytes(wav_bytes(300))
            socket.receive_json()

    session.expire_all()
    call = session.execute(select(Call)).scalar_one()
    assert call.ended_at is not None


def test_the_harness_permits_turns_up_to_the_configured_limit(
    harness, open_weekdays, haircut
) -> None:
    client, model = harness(say("One."), say("Two."), max_harness_turns=2)

    with client.websocket_connect("/ws/harness") as socket:
        socket.receive_json()
        socket.receive_bytes()
        for _ in range(2):
            socket.send_bytes(wav_bytes(300))
            turn = socket.receive_json()
            socket.receive_bytes()
            assert turn["failed"] is False

    assert model.call_count == 2


def test_the_harness_closes_after_the_turn_limit(
    harness, open_weekdays, haircut
) -> None:
    """A third utterance must not reach the model at all: `FakeModel` would
    raise `AssertionError` if it were asked for a response that was never
    scripted, which is exactly what a limit that let a third turn through
    would cause."""
    client, model = harness(say("One."), say("Two."), max_harness_turns=2)

    with pytest.raises(starlette.websockets.WebSocketDisconnect) as disconnected:
        with client.websocket_connect("/ws/harness") as socket:
            socket.receive_json()
            socket.receive_bytes()
            for _ in range(2):
                socket.send_bytes(wav_bytes(300))
                socket.receive_json()
                socket.receive_bytes()
            socket.send_bytes(wav_bytes(300))
            socket.receive_json()

    assert disconnected.value.code == 1000
    assert model.call_count == 2


def test_a_turn_with_a_tool_call_still_counts_as_one_turn(
    harness, open_weekdays, haircut
) -> None:
    """One user utterance, two model calls inside it (a tool round trip),
    must count once against `max_harness_turns` — not twice. Counted at the
    completed-turn point in `app/api/harness.py`, never per model/tool
    iteration, which `max_tool_iterations` already bounds on its own."""
    client, model = harness(
        use_tools(
            ("check_availability", {"service_name": "Haircut", "day": "2026-03-02"})
        ),
        say("Nine, half nine or ten."),
        max_harness_turns=1,
    )

    with pytest.raises(starlette.websockets.WebSocketDisconnect) as disconnected:
        with client.websocket_connect("/ws/harness") as socket:
            socket.receive_json()
            socket.receive_bytes()

            socket.send_bytes(wav_bytes(300))
            turn = socket.receive_json()
            socket.receive_bytes()
            assert turn["failed"] is False
            assert model.call_count == 2  # the tool round trip, one turn

            # The one allowed turn is already spent, whatever it cost inside.
            socket.send_bytes(wav_bytes(300))
            socket.receive_json()

    assert disconnected.value.code == 1000


def test_twenty_turns_is_the_harness_default() -> None:
    assert Settings(_env_file=None).max_harness_turns == 20
