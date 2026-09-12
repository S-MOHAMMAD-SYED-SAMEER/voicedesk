"""A realtime phone call over a simulated carrier stream.

Simulated, and said so: a real WebSocket against the real application and a
real database, with scripted model and speech providers. No telephone call is
placed and no carrier behaviour is proved.
"""

import base64
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.audio.telephony import mulaw_encode
from app.models import Call, Turn
from app.providers.offline_streaming import (
    OfflineStreamingSpeechToText,
    OfflineStreamingTextToSpeech,
)

from .conftest import (
    TWILIO_CALL_SID,
    TWILIO_STREAM_SID,
    FakeModel,
    pcm_silence,
    pcm_tone,
    say,
    start_frame,
    twilio_frame,
    use_tools,
)

STREAM = "/telephony/stream"
LOUD = mulaw_encode(pcm_tone(9000))
QUIET = mulaw_encode(pcm_silence())
MONDAY = "2026-03-02"


def _media(audio: bytes, stream_sid: str = TWILIO_STREAM_SID) -> str:
    return twilio_frame(
        "media",
        streamSid=stream_sid,
        media={"track": "inbound", "payload": base64.b64encode(audio).decode()},
    )


@pytest.fixture
def realtime_telephony(
    migrated_engine: Engine, telephony_settings, monkeypatch: pytest.MonkeyPatch
):
    """A client whose telephony stream runs the realtime path."""
    from app.main import create_app

    def build(*responses, stt=None, tts=None, model=None, raises=None, **overrides):
        fields = {"realtime_enabled": True}
        fields.update(overrides)
        settings = telephony_settings.model_copy(update=fields)
        scripted = model or FakeModel(*responses, raises=raises)
        monkeypatch.setattr("app.telephony.stream.get_settings", lambda: settings)
        monkeypatch.setattr(
            "app.telephony.stream.build_model", lambda settings=None: scripted
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_streaming_stt",
            lambda settings=None: stt or OfflineStreamingSpeechToText(),
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_streaming_tts",
            lambda settings=None: tts
            or OfflineStreamingTextToSpeech(
                sample_rate=8000, chunk_ms=200, ms_per_character=2
            ),
        )
        return TestClient(create_app()), scripted

    return build


def _collect(socket, limit: int = 400) -> tuple[list[dict], list[str]]:
    """Read frames until the socket runs dry, sorting them by kind."""
    media: list[dict] = []
    events: list[str] = []
    for _ in range(limit):
        try:
            message = socket.receive_json()
        except Exception:
            break
        events.append(message["event"])
        if message["event"] == "media":
            media.append(message)
        if message["event"] in {"mark", "clear"}:
            break
    return media, events


def _speak(socket, frames: int = 10) -> None:
    for _ in range(frames):
        socket.send_text(_media(LOUD))


def _fall_silent(socket, frames: int = 40) -> None:
    for _ in range(frames):
        socket.send_text(_media(QUIET))


# --- the realtime path ----------------------------------------------------


def test_a_start_greets_the_caller_in_realtime(
    realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    tts = OfflineStreamingTextToSpeech(
        sample_rate=8000, chunk_ms=200, ms_per_character=2
    )
    client, model = realtime_telephony(say("Certainly."), tts=tts)

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        media, events = _collect(socket)

    assert media
    assert events[-1] == "mark"
    assert tts.spoken == ["Thanks for calling. How can I help?"]
    assert model.call_count == 0


def test_speech_then_silence_produces_one_turn(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, model = realtime_telephony(say("We're open until five."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        _speak(socket)
        _fall_silent(socket)
        media, events = _collect(socket)

    assert model.call_count == 1
    assert media
    turns = db_session.execute(select(Turn).order_by(Turn.created_at)).scalars().all()
    assert [turn.text for turn in turns] == [
        "I'd like to book an appointment",
        "We're open until five.",
    ]


def test_the_reader_keeps_reading_while_a_turn_runs(
    realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The previous milestone was deaf for the length of a turn.

    Frames sent immediately after the utterance are accepted rather than
    queueing behind the reply — which is the whole basis of barge-in.
    """
    client, model = realtime_telephony(say("One."), say("Two."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        _speak(socket)
        _fall_silent(socket)
        _speak(socket)
        _fall_silent(socket)
        _collect(socket, limit=1000)

    assert model.call_count == 2


def test_outbound_audio_is_carrier_sized(
    realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = realtime_telephony(say("We're open until five."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        media, _ = _collect(socket)

    payloads = [base64.b64decode(frame["media"]["payload"]) for frame in media]
    assert payloads
    assert all(len(payload) <= 160 for payload in payloads)
    assert all(len(payload) == 160 for payload in payloads[:-1])


def test_a_mark_records_when_playback_finished(
    realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = realtime_telephony(say("Certainly."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        media, events = _collect(socket)
        socket.send_text(
            twilio_frame(
                "mark", streamSid=TWILIO_STREAM_SID, mark={"name": "reply-1"}
            )
        )
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert "mark" in events


def test_a_booking_made_by_telephone_in_realtime_reaches_the_database(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    from app.models import Appointment

    client, _ = realtime_telephony(
        use_tools(("check_availability", {"service_name": "Haircut", "day": MONDAY})),
        use_tools(
            (
                "book_appointment",
                {
                    "service_name": "Haircut",
                    "starts_at": "2026-03-02T10:00:00+00:00",
                    "customer_name": "Ada Lovelace",
                    "phone": "+447700900123",
                },
            )
        ),
        say("You're booked in for ten."),
    )

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        _speak(socket)
        _fall_silent(socket)
        _collect(socket, limit=1000)

    appointment = db_session.execute(select(Appointment)).scalar_one()
    assert appointment.call_id == twilio_call.id


def test_latency_is_written_onto_the_turn_rows(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The columns that have been null since milestone 1."""
    client, _ = realtime_telephony(say("We're open until five."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        _speak(socket)
        _fall_silent(socket)
        _collect(socket, limit=1000)

    turns = db_session.execute(select(Turn).order_by(Turn.created_at)).scalars().all()
    caller, agent = turns[0], turns[1]
    assert caller.audio_ms is not None and caller.audio_ms > 0
    assert agent.stt_latency_ms is not None
    assert agent.tts_latency_ms is not None
    assert agent.llm_latency_ms is not None


# --- what does not change -------------------------------------------------


def test_silence_alone_produces_no_turn(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, model = realtime_telephony(say("should never be used"))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        _fall_silent(socket, frames=120)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert model.call_count == 0
    assert db_session.execute(select(Turn)).first() is None


def test_our_own_voice_is_never_transcribed(
    realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, model = realtime_telephony(say("should never be used"))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        for _ in range(30):
            socket.send_text(
                twilio_frame(
                    "media",
                    streamSid=TWILIO_STREAM_SID,
                    media={
                        "track": "outbound",
                        "payload": base64.b64encode(LOUD).decode(),
                    },
                )
            )
        _fall_silent(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert model.call_count == 0


def test_a_malformed_frame_does_not_end_a_realtime_call(
    realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = realtime_telephony(say("We're open until five."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        socket.send_text("not json at all")
        _speak(socket)
        _fall_silent(socket)
        media, _ = _collect(socket, limit=1000)

    assert media


def test_a_stop_during_a_realtime_call_ends_it(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = realtime_telephony(say("Certainly."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    db_session.refresh(twilio_call)
    assert twilio_call.ended_at is not None
    assert twilio_call.outcome is None
    assert twilio_call.total_cost_usd is None


def test_a_disconnect_during_a_realtime_call_ends_it(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = realtime_telephony(say("Certainly."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _collect(socket)

    db_session.refresh(twilio_call)
    assert twilio_call.ended_at is not None


def test_two_realtime_calls_keep_their_own_state(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    second = Call(
        direction=twilio_call.direction,
        from_number="+440000000003",
        to_number="+440000000004",
        provider_call_sid="CA-second-call",
    )
    db_session.add(second)
    db_session.commit()

    client, _ = realtime_telephony(say("First."), say("Second."))

    with client.websocket_connect(STREAM) as first_socket:
        first_socket.send_text(start_frame())
        _collect(first_socket)
        with client.websocket_connect(STREAM) as second_socket:
            second_socket.send_text(
                start_frame(call_sid="CA-second-call", stream_sid="MZ-second")
            )
            _collect(second_socket)

    db_session.refresh(second)
    assert second.ended_at is not None


# --- the milestone-6 path is still there ----------------------------------


def test_with_realtime_off_the_utterance_path_runs(
    db_session: Session, realtime_telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The default, and what the earlier milestone's tests still exercise."""
    from app.providers.offline_stt import OfflineSpeechToText
    from app.providers.offline_tts import OfflineTextToSpeech

    client, model = realtime_telephony(say("Certainly."), realtime_enabled=False)

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        media, events = _collect(socket)

    assert media
    assert events[-1] == "mark"
