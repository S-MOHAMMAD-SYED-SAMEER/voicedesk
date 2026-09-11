"""The browser harness: one WebSocket, one call, whole utterances.

The model, transcriber and synthesiser are all replaced, so nothing here needs
a key, a network or a microphone.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.api.harness import BROWSER_FROM_NUMBER, BROWSER_TO_NUMBER, GREETING
from app.models import Appointment, Call, Turn
from app.providers.speech import read_wav

from .conftest import FakeModel, FakeSTT, FakeTTS, say, use_tools, wav_bytes

MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"


def _check():
    return ("check_availability", {"service_name": "Haircut", "day": MONDAY})


def _book():
    return (
        "book_appointment",
        {
            "service_name": "Haircut",
            "starts_at": TEN,
            "customer_name": "Ada Lovelace",
            "phone": "+447700900123",
        },
    )


@pytest.fixture
def harness(migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch):
    """A `TestClient` whose harness uses scripted providers."""
    from app.main import create_app

    def build(*responses, stt=None, tts=None, model=None, raises=None):
        scripted = model or FakeModel(*responses, raises=raises)
        monkeypatch.setattr(
            "app.api.harness.build_model", lambda settings=None: scripted
        )
        monkeypatch.setattr(
            "app.api.harness.build_stt", lambda settings=None: stt or FakeSTT()
        )
        monkeypatch.setattr(
            "app.api.harness.build_tts", lambda settings=None: tts or FakeTTS()
        )
        return TestClient(create_app()), scripted

    return build


def _ready(socket) -> dict:
    return socket.receive_json()


# --- connecting -----------------------------------------------------------


def test_connecting_answers_with_ready_and_a_call(
    session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("Certainly."))

    with client.websocket_connect("/ws/harness") as socket:
        ready = _ready(socket)
        socket.receive_bytes()  # the greeting audio

    call = session.execute(select(Call)).scalar_one()
    assert ready["type"] == "ready"
    assert ready["call_id"] == str(call.id)
    assert ready["greeting"] == GREETING


def test_the_browser_call_uses_sentinels_not_invented_numbers(
    session: Session, harness, open_weekdays, haircut
) -> None:
    """A fake telephone number would eventually be read as a real one."""
    client, _ = harness(say("Certainly."))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()

    call = session.execute(select(Call)).scalar_one()
    assert call.from_number == BROWSER_FROM_NUMBER == "browser"
    assert call.to_number == BROWSER_TO_NUMBER == "harness"


def test_ready_describes_the_audio_the_server_will_accept(
    harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("Certainly."))

    with client.websocket_connect("/ws/harness") as socket:
        ready = _ready(socket)
        socket.receive_bytes()

    assert ready["format"] == {
        "encoding": "pcm_s16le",
        "sample_rate": 16000,
        "channels": 1,
        "container": "wav",
    }
    assert ready["max_utterance_bytes"] == 1_048_576


def test_the_greeting_audio_is_playable_and_is_not_a_turn(
    session: Session, harness, open_weekdays, haircut
) -> None:
    """Picking up the phone is not something anyone said."""
    tts = FakeTTS()
    client, model = harness(say("Certainly."), tts=tts)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        audio = socket.receive_bytes()

    assert read_wav(audio).format.sample_rate == 16000
    assert tts.calls == [GREETING]
    assert model.call_count == 0
    assert session.execute(select(Turn)).first() is None


# --- a turn ----------------------------------------------------------------


def test_an_utterance_comes_back_as_json_then_audio(
    harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("We're open until five."), stt=FakeSTT("When do you close?"))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(wav_bytes(500))
        turn = socket.receive_json()
        audio = socket.receive_bytes()

    assert turn["type"] == "turn"
    assert turn["transcript"] == "When do you close?"
    assert turn["reply"] == "We're open until five."
    assert turn["failed"] is False
    assert turn["audio_bytes"] == len(audio)
    assert read_wav(audio).format.channels == 1


def test_the_turn_message_reports_timings_and_tools(
    harness, open_weekdays, haircut
) -> None:
    client, _ = harness(use_tools(_check()), say("Nine, half nine or ten."))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(wav_bytes(500))
        turn = socket.receive_json()
        socket.receive_bytes()

    assert set(turn["timings"]) == {"stt_ms", "llm_ms", "tts_ms", "total_ms"}
    assert turn["tool_calls"] == [
        {"name": "check_availability", "success": True}
    ]


def test_several_turns_share_one_call_and_one_conversation(
    session: Session, harness, open_weekdays, haircut
) -> None:
    client, model = harness(
        say("Of course, what day?"), say("Nine or ten."), say("Booked.")
    )

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        for _ in range(3):
            socket.send_bytes(wav_bytes(300))
            socket.receive_json()
            socket.receive_bytes()

    assert len(session.execute(select(Call)).scalars().all()) == 1
    assert len(session.execute(select(Turn)).scalars().all()) == 6
    # History grew across the turns rather than restarting each time.
    assert len(model.requests[-1]["messages"]) == 5


def test_a_booking_made_over_the_socket_reaches_the_database(
    session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(
        use_tools(_check()), use_tools(_book()), say("You're booked in for ten.")
    )

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    appointment = session.execute(select(Appointment)).scalar_one()
    call = session.execute(select(Call)).scalar_one()
    assert appointment.call_id == call.id


# --- bad input --------------------------------------------------------------


def test_audio_that_is_not_a_wav_is_refused_without_a_turn(
    session: Session, harness, open_weekdays, haircut
) -> None:
    client, model = harness(say("should never be used"))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(b"\x00" * 2048)
        message = socket.receive_json()

    assert message["type"] == "error"
    assert message["reason"] == "invalid_audio"
    assert model.call_count == 0
    assert session.execute(select(Turn)).first() is None


def test_stereo_audio_is_refused(harness, open_weekdays, haircut) -> None:
    client, _ = harness(say("should never be used"))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(wav_bytes(300, channels=2))
        message = socket.receive_json()

    assert message["reason"] == "invalid_audio"
    assert "mono" in message["detail"]


def test_an_oversized_utterance_is_refused(harness, open_weekdays, haircut) -> None:
    client, model = harness(say("should never be used"))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(wav_bytes(40_000))  # ~1.28 MB, over the 1 MiB cap
        message = socket.receive_json()

    assert message["reason"] == "invalid_audio"
    assert "at most" in message["detail"]
    assert model.call_count == 0


def test_a_refused_frame_does_not_end_the_call(harness, open_weekdays, haircut) -> None:
    client, _ = harness(say("We're open until five."))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(b"rubbish")
        assert socket.receive_json()["type"] == "error"

        socket.send_bytes(wav_bytes(300))
        assert socket.receive_json()["type"] == "turn"
        socket.receive_bytes()


def test_an_unexpected_text_frame_is_reported(harness, open_weekdays, haircut) -> None:
    client, _ = harness(say("Certainly."))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_text(json.dumps({"type": "something else"}))
        message = socket.receive_json()

    assert message["reason"] == "unexpected_text_frame"


def test_a_provider_that_blows_up_is_reported_not_dropped(
    harness, open_weekdays, haircut
) -> None:
    """A misconfigured provider is a harness message, not a dead socket."""

    class Exploding:
        def transcribe(self, audio):
            raise TypeError("Could not resolve authentication method.")

    client, _ = harness(say("Certainly."), stt=Exploding())

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(wav_bytes(300))
        message = socket.receive_json()

    assert message["reason"] == "turn_failed"
    assert "TypeError" in message["detail"]


# --- hanging up -------------------------------------------------------------


def test_hanging_up_records_when_the_call_ended(
    session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("Certainly."))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_text(json.dumps({"type": "hangup"}))

    call = session.execute(select(Call)).scalar_one()
    assert call.ended_at is not None
    assert call.ended_at >= call.started_at


def test_disconnecting_records_when_the_call_ended(
    session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("Certainly."))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()

    call = session.execute(select(Call)).scalar_one()
    assert call.ended_at is not None


def test_the_outcome_is_left_for_the_milestone_that_owns_it(
    session: Session, harness, open_weekdays, haircut
) -> None:
    """A guess written now would be indistinguishable from a fact later."""
    client, _ = harness(
        use_tools(_check()), use_tools(_book()), say("You're booked in.")
    )

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.receive_bytes()
        socket.send_bytes(wav_bytes(300))
        socket.receive_json()
        socket.receive_bytes()
        socket.send_text(json.dumps({"type": "hangup"}))

    assert session.execute(select(Call)).scalar_one().outcome is None


# --- the page ---------------------------------------------------------------


def test_the_page_is_served(harness, open_weekdays, haircut) -> None:
    client, _ = harness(say("Certainly."))

    response = client.get("/harness")

    assert response.status_code == 200
    assert "/ws/harness" in response.text
    assert "text/html" in response.headers["content-type"]


def test_health_is_unchanged(harness, open_weekdays, haircut) -> None:
    client, _ = harness(say("Certainly."))

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert set(body) == {"status", "app", "version", "environment"}
