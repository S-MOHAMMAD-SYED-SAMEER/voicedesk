"""The browser harness in realtime mode, and the push-to-talk fallback."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.models import Call, Turn
from app.providers.offline_streaming import (
    OfflineStreamingSpeechToText,
    OfflineStreamingTextToSpeech,
)

from .conftest import FakeModel, FakeSTT, FakeTTS, pcm_silence, pcm_tone, say, wav_bytes

LOUD = pcm_tone(9000, sample_rate=16000)
QUIET = pcm_silence(sample_rate=16000)


@pytest.fixture
def harness(migrated_engine: Engine, calendar_settings, monkeypatch):
    """A client whose harness uses scripted providers, realtime or not."""
    from app.main import create_app

    def build(*responses, model=None, stt=None, tts=None, raises=None, **overrides):
        settings = calendar_settings.model_copy(update=overrides)
        scripted = model or FakeModel(*responses, raises=raises)
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
        monkeypatch.setattr(
            "app.api.harness.build_streaming_stt",
            lambda settings=None: stt or OfflineStreamingSpeechToText(),
        )
        monkeypatch.setattr(
            "app.api.harness.build_streaming_tts",
            lambda settings=None: tts
            or OfflineStreamingTextToSpeech(
                sample_rate=16000, chunk_ms=200, ms_per_character=2
            ),
        )
        return TestClient(create_app()), scripted

    return build


def _ready(socket) -> dict:
    message = socket.receive_json()
    socket.receive_bytes()  # the greeting audio
    return message


def _speak(socket, frames: int = 20) -> None:
    for _ in range(frames):
        socket.send_bytes(LOUD)


def _fall_silent(socket, frames: int = 40) -> None:
    for _ in range(frames):
        socket.send_bytes(QUIET)


def _drain(socket, limit: int = 400) -> tuple[list[dict], int]:
    """Read until the turn arrives, sorting JSON messages from audio frames.

    The turn message is sent by the turn task as soon as it finishes, so this
    is bounded by the turn completing rather than by a timeout.
    """
    messages: list[dict] = []
    audio = 0
    for _ in range(limit):
        frame = socket.receive()
        if frame.get("type") == "websocket.disconnect":
            break
        if frame.get("text") is not None:
            body = json.loads(frame["text"])
            messages.append(body)
            if body.get("type") == "turn":
                break
        elif frame.get("bytes") is not None:
            audio += 1
    return messages, audio


# --- realtime -------------------------------------------------------------


def test_ready_says_which_mode_the_call_is_in(
    harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("Certainly."), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        ready = _ready(socket)

    assert ready["realtime"] is True
    assert ready["frame_ms"] == 20


def test_continuous_frames_produce_a_turn(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, model = harness(say("We're open until five."), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        _speak(socket)
        _fall_silent(socket)
        messages, audio = _drain(socket)

    assert model.call_count == 1
    assert audio > 0
    turns = db_session.execute(select(Turn).order_by(Turn.created_at)).scalars().all()
    assert [turn.text for turn in turns] == [
        "I'd like to book an appointment",
        "We're open until five.",
    ]


def test_partial_transcripts_are_shown(harness, open_weekdays, haircut) -> None:
    client, _ = harness(say("Certainly."), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        _speak(socket)
        _fall_silent(socket)
        messages, _ = _drain(socket)

    partials = [m["text"] for m in messages if m["type"] == "partial"]
    assert partials == ["I'd like", "I'd like to book"]


def test_a_partial_never_reaches_the_model(
    harness, open_weekdays, haircut
) -> None:
    client, model = harness(say("Certainly."), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        _speak(socket)
        _fall_silent(socket)
        _drain(socket)

    asked = [
        message.content[0]["text"]
        for message in model.requests[0]["messages"]
        if message.role == "user"
    ]
    assert asked == ["I'd like to book an appointment"]


def test_the_turn_message_reports_what_happened(
    harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("We're open until five."), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        _speak(socket)
        _fall_silent(socket)
        messages, _ = _drain(socket)

    turn = [m for m in messages if m["type"] == "turn"][0]
    assert turn["transcript"] == "I'd like to book an appointment"
    assert turn["reply"] == "We're open until five."
    assert turn["failed"] is False
    assert set(turn["timings"]) == {"stt_ms", "llm_ms", "tts_ms", "total_ms"}


def test_silence_alone_produces_nothing(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, model = harness(say("should never be used"), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        _fall_silent(socket, frames=120)
        socket.send_text(json.dumps({"type": "hangup"}))

    assert model.call_count == 0
    assert db_session.execute(select(Turn)).first() is None


def test_a_disconnect_ends_the_call(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("Certainly."), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)

    call = db_session.execute(select(Call)).scalar_one()
    assert call.ended_at is not None
    assert call.outcome is None


def test_latency_is_written_onto_the_turn_rows(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("We're open until five."), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        _speak(socket)
        _fall_silent(socket)
        _drain(socket)

    turns = db_session.execute(select(Turn).order_by(Turn.created_at)).scalars().all()
    assert turns[0].audio_ms is not None
    assert turns[1].stt_latency_ms is not None
    assert turns[1].tts_latency_ms is not None


# --- switching modes ------------------------------------------------------


def test_the_mode_can_be_switched_mid_call(harness, open_weekdays, haircut) -> None:
    """So the two paths can be compared without restarting anything."""
    client, _ = harness(say("Certainly."), realtime_enabled=False)

    with client.websocket_connect("/ws/harness") as socket:
        ready = _ready(socket)
        socket.send_text(json.dumps({"type": "mode", "value": "realtime"}))
        switched = socket.receive_json()

    assert ready["realtime"] is False
    assert switched == {"type": "mode", "realtime": True}


# --- press-to-talk still works --------------------------------------------


def test_push_to_talk_is_unchanged_when_realtime_is_off(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, model = harness(say("We're open until five."), realtime_enabled=False)

    with client.websocket_connect("/ws/harness") as socket:
        ready = _ready(socket)
        socket.send_bytes(wav_bytes(500))
        turn = socket.receive_json()
        socket.receive_bytes()

    assert ready["realtime"] is False
    assert turn["type"] == "turn"
    assert turn["reply"] == "We're open until five."
    assert model.call_count == 1


def test_push_to_talk_still_refuses_bad_audio(
    harness, open_weekdays, haircut
) -> None:
    client, _ = harness(say("Certainly."), realtime_enabled=False)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(b"\x00" * 2048)
        message = socket.receive_json()

    assert message["reason"] == "invalid_audio"


def test_the_page_offers_both_modes(harness, open_weekdays, haircut) -> None:
    client, _ = harness(say("Certainly."))

    body = client.get("/harness").text

    assert "Hold to talk" in body
    assert "Realtime:" in body
    assert "stopPlayback" in body
