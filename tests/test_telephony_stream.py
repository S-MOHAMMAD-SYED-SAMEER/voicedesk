"""A whole phone call over a simulated Twilio media stream.

Simulated, and said so: these drive a real WebSocket against the real
application and a real database, with scripted model and speech providers.
They do not place a telephone call and prove nothing about a carrier's
behaviour on the public network.
"""

import base64
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.audio.telephony import FRAME_BYTES, mulaw_decode
from app.models import Appointment, Call, ToolCall, Turn
from app.providers.tts import VoiceUnavailable

from .conftest import (
    TWILIO_CALL_SID,
    TWILIO_STREAM_SID,
    FakeModel,
    FakeSTT,
    FakeTTS,
    media_text_frame,
    mulaw_frame,
    say,
    start_frame,
    twilio_frame,
    use_tools,
)

STREAM = "/telephony/stream"
MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"
# Comfortably above the configured threshold of 500, and comfortably below it.
LOUD = mulaw_frame(9000)
QUIET = mulaw_frame(0)


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
def telephony(
    migrated_engine: Engine, telephony_settings, monkeypatch: pytest.MonkeyPatch
):
    """A client whose telephony stream uses scripted providers."""
    from app.main import create_app

    def build(*responses, stt=None, tts=None, model=None, raises=None, **overrides):
        settings = (
            telephony_settings.model_copy(update=overrides)
            if overrides
            else telephony_settings
        )
        scripted = model or FakeModel(*responses, raises=raises)
        monkeypatch.setattr("app.telephony.stream.get_settings", lambda: settings)
        monkeypatch.setattr(
            "app.telephony.stream.build_model", lambda settings=None: scripted
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_stt", lambda settings=None: stt or FakeSTT()
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_tts", lambda settings=None: tts or FakeTTS()
        )
        return TestClient(create_app()), scripted

    return build


def _drain_audio(socket) -> tuple[list[bytes], dict]:
    """Read media frames until the mark that ends the batch."""
    payloads: list[bytes] = []
    while True:
        message = socket.receive_json()
        if message["event"] == "media":
            payloads.append(base64.b64decode(message["media"]["payload"]))
            continue
        return payloads, message


def _speak(socket, frames: int = 3) -> None:
    for _ in range(frames):
        socket.send_text(media_text_frame(LOUD))


def _fall_silent(socket, frames: int = 50) -> None:
    """Enough quiet frames to cross the 800 ms boundary."""
    for _ in range(frames):
        socket.send_text(media_text_frame(QUIET))


# --- starting -------------------------------------------------------------


def test_a_start_greets_the_caller(telephony, twilio_call, open_weekdays, haircut) -> None:
    tts = FakeTTS()
    client, model = telephony(say("Certainly."), tts=tts)

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(twilio_frame("connected", protocol="Call", version="1.0.0"))
        socket.send_text(start_frame())
        payloads, mark = _drain_audio(socket)

    assert payloads, "the greeting should have been played"
    assert mark["event"] == "mark"
    assert tts.calls == ["Thanks for calling. How can I help?"]
    assert model.call_count == 0


def test_the_greeting_creates_no_transcript_and_calls_no_tool(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """Picking up the phone is not something anybody said."""
    client, _ = telephony(say("Certainly."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)

    assert session.execute(select(Turn)).first() is None
    assert session.execute(select(ToolCall)).first() is None


def test_an_unknown_call_is_refused(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The webhook is signed; a stream naming a call nobody answered is not one.

    No row is invented for it, because being able to find the call is the only
    thing authenticating this socket.
    """
    from starlette.websockets import WebSocketDisconnect

    client, model = telephony(say("Certainly."))

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(STREAM) as socket:
            socket.send_text(start_frame(call_sid="CA-never-answered"))
            socket.receive_json()

    calls = session.execute(select(Call)).scalars().all()
    assert [call.provider_call_sid for call in calls] == [TWILIO_CALL_SID]
    assert model.call_count == 0
    assert session.execute(select(Turn)).first() is None


@pytest.mark.parametrize(
    "media_format",
    [{"encoding": "audio/l16"}, {"sampleRate": 16000}, {"channels": 2}],
    ids=["encoding", "rate", "channels"],
)
def test_audio_this_system_cannot_read_is_refused(
    telephony, twilio_call, open_weekdays, haircut, media_format
) -> None:
    from starlette.websockets import WebSocketDisconnect

    client, _ = telephony(say("Certainly."))

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(STREAM) as socket:
            socket.send_text(start_frame(**media_format))
            socket.receive_json()


def test_a_second_start_does_not_repoint_the_call(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """Whatever that is, it is not this call becoming another one."""
    other = Call(
        direction=twilio_call.direction,
        from_number="+440000000001",
        to_number="+440000000002",
        provider_call_sid="CA-another-call",
    )
    session.add(other)
    session.commit()

    client, _ = telephony(say("Certainly."))
    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        socket.send_text(start_frame(call_sid="CA-another-call"))
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    session.refresh(other)
    assert other.ended_at is None
    session.refresh(twilio_call)
    assert twilio_call.ended_at is not None


def test_the_stream_is_closed_when_telephony_is_disabled(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    from starlette.websockets import WebSocketDisconnect

    client, _ = telephony(say("Certainly."), telephony_enabled=False)

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(STREAM) as socket:
            socket.receive_json()


# --- a turn ---------------------------------------------------------------


def test_speech_then_silence_produces_one_turn(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, model = telephony(
        say("We're open until five."), stt=FakeSTT("When do you close?")
    )

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        _speak(socket)
        _fall_silent(socket)
        payloads, mark = _drain_audio(socket)

    assert model.call_count == 1
    assert mark["event"] == "mark"
    turns = session.execute(select(Turn).order_by(Turn.created_at)).scalars().all()
    assert [turn.text for turn in turns] == [
        "When do you close?",
        "We're open until five.",
    ]


def test_the_reply_goes_out_as_carrier_sized_frames(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = telephony(say("We're open until five."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        _speak(socket)
        _fall_silent(socket)
        payloads, _ = _drain_audio(socket)

    assert payloads
    assert all(len(payload) <= FRAME_BYTES for payload in payloads)
    assert all(len(payload) == FRAME_BYTES for payload in payloads[:-1])
    # And it really is µ-law: decoding gives whole 16-bit samples.
    assert len(mulaw_decode(b"".join(payloads))) == 2 * sum(map(len, payloads))


def test_silence_alone_produces_no_turn(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """A caller who says nothing is not a question for a model."""
    client, model = telephony(say("should never be used"))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        _fall_silent(socket, frames=120)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert model.call_count == 0
    assert session.execute(select(Turn)).first() is None


def test_a_caller_who_never_pauses_is_still_answered(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The maximum-utterance ceiling is also what bounds the buffer.

    Ten 20 ms frames is the 200 ms ceiling exactly, so the utterance closes
    without anybody having stopped talking.
    """
    client, model = telephony(
        say("Let me stop you there."), telephony_max_utterance_ms=200
    )

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        _speak(socket, frames=10)
        payloads, mark = _drain_audio(socket)

    assert model.call_count == 1
    assert payloads and mark["event"] == "mark"


def test_several_turns_share_one_call_and_one_conversation(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, model = telephony(say("Of course."), say("Nine or ten."), say("Booked."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        for _ in range(3):
            _speak(socket)
            _fall_silent(socket)
            _drain_audio(socket)

    assert len(session.execute(select(Call)).scalars().all()) == 1
    assert len(session.execute(select(Turn)).scalars().all()) == 6
    assert len(model.requests[-1]["messages"]) == 5


def test_a_booking_made_by_telephone_reaches_the_database(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = telephony(
        use_tools(_check()), use_tools(_book()), say("You're booked in for ten.")
    )

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        _speak(socket)
        _fall_silent(socket)
        _drain_audio(socket)

    appointment = session.execute(select(Appointment)).scalar_one()
    assert appointment.call_id == twilio_call.id


# --- audio that should not reach the dialogue -----------------------------


def test_our_own_voice_is_never_transcribed(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    """Feeding the outbound track back to speech recognition is a loop."""
    client, model = telephony(say("should never be used"))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        for _ in range(30):
            socket.send_text(media_text_frame(LOUD, track="outbound"))
        _fall_silent(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert model.call_count == 0


def test_audio_arriving_before_the_call_is_identified_is_dropped(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, model = telephony(say("should never be used"))

    with client.websocket_connect(STREAM) as socket:
        _speak(socket, frames=30)
        _fall_silent(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert model.call_count == 0
    assert session.execute(select(Turn)).first() is None


def test_audio_for_another_stream_is_ignored(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, model = telephony(say("should never be used"))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        for _ in range(30):
            socket.send_text(media_text_frame(LOUD, stream_sid="MZ-somebody-else"))
        _fall_silent(socket, frames=1)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert model.call_count == 0


# --- frames that make no sense ---------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        "not json at all",
        "{",
        json.dumps({"streamSid": "MZ1"}),
        json.dumps({"event": "media", "streamSid": "MZ1"}),
        json.dumps({"event": "media", "streamSid": "MZ1", "media": {"payload": "!!"}}),
        json.dumps({"event": "dtmf", "streamSid": "MZ1", "dtmf": {"digit": "1"}}),
        json.dumps({"event": "mark", "streamSid": "MZ1", "mark": {"name": "x"}}),
    ],
    ids=["text", "truncated", "no-event", "no-media", "bad-base64", "dtmf", "mark"],
)
def test_one_bad_frame_does_not_end_the_call(
    telephony, twilio_call, open_weekdays, haircut, frame: str
) -> None:
    """A carrier is entitled to send something this milestone never heard of."""
    client, _ = telephony(say("We're open until five."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)

        socket.send_text(frame)

        _speak(socket)
        _fall_silent(socket)
        payloads, mark = _drain_audio(socket)

    assert payloads
    assert mark["event"] == "mark"


def test_a_binary_frame_is_ignored(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The carrier's control channel is text; binary is not in this protocol."""
    client, _ = telephony(say("We're open until five."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        socket.send_bytes(b"\x00\x01\x02")
        _speak(socket)
        _fall_silent(socket)
        payloads, _ = _drain_audio(socket)

    assert payloads


def test_a_provider_that_blows_up_does_not_end_the_call(
    telephony, twilio_call, open_weekdays, haircut
) -> None:
    class Exploding:
        def transcribe(self, audio):
            raise TypeError("Could not resolve authentication method.")

    client, _ = telephony(say("Certainly."), stt=Exploding())

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        _speak(socket)
        _fall_silent(socket)
        # No reply is played, and the socket is still usable.
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))


def test_a_synthesiser_failure_leaves_the_dialogue_intact(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """The booking happened. Retrying for audio would risk a second one."""
    client, _ = telephony(
        use_tools(_check()),
        use_tools(_book()),
        say("You're booked in."),
        tts=FakeTTS(raises=VoiceUnavailable("down")),
    )

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _speak(socket)
        _fall_silent(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    assert len(session.execute(select(Appointment)).scalars().all()) == 1
    assert len(session.execute(select(Turn)).scalars().all()) == 2


# --- ending ---------------------------------------------------------------


def test_a_stop_records_when_the_call_ended(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = telephony(say("Certainly."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        socket.send_text(
            twilio_frame("stop", streamSid=TWILIO_STREAM_SID, stop={"callSid": TWILIO_CALL_SID})
        )

    session.refresh(twilio_call)
    assert twilio_call.ended_at is not None
    assert twilio_call.ended_at >= twilio_call.started_at


def test_a_disconnect_records_when_the_call_ended(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """Networks drop without a stop event; the call still ended."""
    client, _ = telephony(say("Certainly."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)

    session.refresh(twilio_call)
    assert twilio_call.ended_at is not None


def test_the_outcome_and_cost_are_left_for_the_milestone_that_owns_them(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = telephony(use_tools(_check()), use_tools(_book()), say("Booked."))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain_audio(socket)
        _speak(socket)
        _fall_silent(socket)
        _drain_audio(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    session.refresh(twilio_call)
    assert twilio_call.outcome is None
    assert twilio_call.total_cost_usd is None


def test_two_calls_keep_their_own_state(
    session: Session, telephony, twilio_call, open_weekdays, haircut
) -> None:
    """Nothing is shared between calls: not the conversation, not the buffer."""
    second = Call(
        direction=twilio_call.direction,
        from_number="+440000000003",
        to_number="+440000000004",
        provider_call_sid="CA-second-call",
    )
    session.add(second)
    session.commit()

    client, model = telephony(say("First."), say("Second."))

    with client.websocket_connect(STREAM) as first_socket:
        first_socket.send_text(start_frame())
        _drain_audio(first_socket)
        _speak(first_socket)
        _fall_silent(first_socket)
        _drain_audio(first_socket)

        with client.websocket_connect(STREAM) as second_socket:
            second_socket.send_text(
                start_frame(call_sid="CA-second-call", stream_sid="MZ-second")
            )
            _drain_audio(second_socket)

    turns = session.execute(select(Turn)).scalars().all()
    assert {turn.call_id for turn in turns} == {twilio_call.id}
    session.refresh(second)
    assert second.ended_at is not None
