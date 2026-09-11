"""The media stream: a telephone on one end, the same receptionist on the other.

    carrier ──► WS /telephony/stream ──► µ-law → Audio ──► VoiceSession
                                                              │
                                    Conversation → tools → calendar → PostgreSQL
                                                              │
    carrier ◄── media frames ◄── µ-law ◄── Speech ◄───────────┘

There is no second dialogue here. `VoiceSession.speak` is the same call the
browser harness makes; only the audio at either end is different, and that
conversion lives in `app/audio/telephony.py`. If this module ever starts
deciding what to say, the layering has gone wrong.

**Two things about this being a real phone line.**

`VoiceSession.speak` is synchronous all the way down — speech recognition, the
model, tool calls, PostgreSQL, synthesis — and running it inline would block
this event loop for seconds while the carrier keeps sending frames. It is
handed to a worker thread instead, so frames continue to be read while a turn
is being worked out.

And a carrier streams continuously, while the dialogue layer wants complete
utterances. The boundary used here is an amplitude timer: buffer once somebody
is speaking, and close the utterance after a fixed stretch below a fixed
threshold. That is **not** voice-activity detection — no spectral analysis, no
adaptive noise floor, no speech classifier, no partial transcripts, no
barge-in. It is the least mechanism that turns a continuous stream into turns,
it is temporary, and the milestone that owns real-time behaviour replaces it.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.audio import VoiceSession, VoiceTurn
from app.audio.telephony import (
    FRAME_BYTES,
    TELEPHONY_SAMPLE_RATE,
    TelephonyAudioError,
    frames,
    mean_amplitude,
    mulaw_decode,
    speech_to_mulaw,
    utterance_from_mulaw,
)
from app.config import Settings, get_settings
from app.db.session import get_sessionmaker
from app.dialogue import Conversation
from app.models import Call
from app.providers.factory import build_model, build_stt, build_tts
from app.telephony.events import (
    ConnectedEvent,
    MalformedEvent,
    MediaEvent,
    StartEvent,
    StopEvent,
    UnknownEvent,
    UnsupportedMediaFormat,
    media_frame,
    mark_frame,
    parse_event,
    require_supported_format,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telephony"])

GREETING = "Thanks for calling. How can I help?"
# Refused the stream. 1008 is "policy violation", which is what an unknown
# call or an audio format we will not decode amounts to.
POLICY_VIOLATION = 1008
# One byte per sample at 8 kHz: eight bytes is a millisecond.
BYTES_PER_MS = TELEPHONY_SAMPLE_RATE // 1000


class Utterance:
    """Continuous audio, cut into turns by an amplitude timer.

    Buffering only starts once something is above the threshold, so a caller
    who says nothing for a minute costs a counter rather than a minute of
    memory. Temporary; see the module docstring.
    """

    def __init__(self, settings: Settings) -> None:
        self._threshold = settings.telephony_silence_threshold
        self._silence_limit_ms = settings.telephony_silence_ms
        self._max_ms = settings.telephony_max_utterance_ms
        self._max_bytes = settings.telephony_max_utterance_ms * BYTES_PER_MS
        self._buffer = bytearray()
        self._silence_ms = 0.0
        self._speaking = False

    @property
    def duration_ms(self) -> float:
        return len(self._buffer) / BYTES_PER_MS

    def add(self, mulaw: bytes) -> bytes | None:
        """One frame in; a complete utterance out, when there is one."""
        if not mulaw:
            return None

        loud = mean_amplitude(mulaw_decode(mulaw)) >= self._threshold
        frame_ms = len(mulaw) / BYTES_PER_MS

        if loud:
            self._speaking = True
            self._silence_ms = 0.0
        elif self._speaking:
            self._silence_ms += frame_ms

        if not self._speaking:
            # Nobody has said anything yet, so there is nothing to keep.
            return None

        self._buffer.extend(mulaw)

        if len(self._buffer) >= self._max_bytes or self.duration_ms >= self._max_ms:
            # A caller who has not paused in thirty seconds still deserves an
            # answer, and an unbounded buffer on an open socket is a bug.
            return self.flush()
        if self._silence_ms >= self._silence_limit_ms:
            return self.flush()
        return None

    def flush(self) -> bytes | None:
        """Whatever has been said so far, and start again."""
        audio = bytes(self._buffer) if self._buffer else None
        self._buffer = bytearray()
        self._silence_ms = 0.0
        self._speaking = False
        return audio


@dataclass
class StreamState:
    """One call's worth of everything, shared with no other call."""

    settings: Settings
    stream_sid: str = ""
    call_sid: str = ""
    call: Call | None = None
    session: Session | None = None
    voice: VoiceSession | None = None
    utterance: Utterance | None = None
    marks: int = field(default=0)

    @property
    def bound(self) -> bool:
        return self.call is not None

    def close(self) -> None:
        """Record that the call ended, and let the database connection go.

        Only `ended_at`. `outcome` and `total_cost_usd` stay null — summarising
        a call belongs to the milestone that owns post-call reporting, and a
        guess written now would be indistinguishable from a fact later. If the
        process dies before this runs, `ended_at` stays null too: nobody hung
        up.
        """
        if self.session is not None:
            try:
                if self.call is not None:
                    self.call.ended_at = datetime.now(UTC)
                    self.session.commit()
            except Exception:  # pragma: no cover - cleanup must not mask errors
                logger.exception("Could not record the end of call %s", self.call_sid)
                self.session.rollback()
            finally:
                self.session.close()
                self.session = None


@router.websocket("/telephony/stream")
async def media_stream(socket: WebSocket) -> None:
    """One telephone call, from the carrier connecting to the caller hanging up."""
    settings = get_settings()
    if not settings.telephony_enabled:
        await socket.close(code=POLICY_VIOLATION)
        return

    await socket.accept()
    state = StreamState(settings=settings)
    try:
        await _converse(socket, state)
    except WebSocketDisconnect:
        pass
    finally:
        state.close()


async def _converse(socket: WebSocket, state: StreamState) -> None:
    """Read frames until the call ends.

    One malformed frame is logged and stepped over. A carrier is entitled to
    send an event this milestone has never heard of, and ending somebody's
    phone call over it would be the worse bug.
    """
    while True:
        message = await socket.receive()
        if message.get("type") == "websocket.disconnect":
            return

        raw = message.get("text")
        if raw is None:
            # The carrier's control channel is text. A binary frame is not
            # something this protocol defines.
            continue

        try:
            event = parse_event(raw)
        except MalformedEvent as exc:
            logger.warning("Ignoring a frame on call %s: %s", state.call_sid, exc)
            continue

        if isinstance(event, ConnectedEvent):
            continue
        if isinstance(event, UnknownEvent):
            logger.info("Ignoring a %r event on call %s", event.name, state.call_sid)
            continue
        if isinstance(event, StartEvent):
            if not await _start(socket, state, event):
                return
            continue
        if isinstance(event, StopEvent):
            return
        if isinstance(event, MediaEvent):
            await _media(socket, state, event)


async def _start(socket: WebSocket, state: StreamState, event: StartEvent) -> bool:
    """Identify the call, build everything it needs, and say hello."""
    if state.bound:
        # A repeated start does not re-point an in-progress call at another
        # one. Whatever this is, it is not this call changing identity.
        logger.warning(
            "Ignoring a second start on call %s (stream %s)",
            state.call_sid,
            event.stream_sid,
        )
        return True

    try:
        require_supported_format(event)
    except UnsupportedMediaFormat as exc:
        logger.error("Refusing stream %s: %s", event.stream_sid, exc)
        await socket.close(code=POLICY_VIOLATION)
        return False

    session = get_sessionmaker()()
    call = session.execute(
        select(Call).where(Call.provider_call_sid == event.call_sid)
    ).scalar_one_or_none()
    if call is None:
        # The webhook creates the call, and the webhook is signed. A stream
        # naming a call nobody answered is not a call: no row is invented for
        # it, because that is the only thing authenticating this socket.
        logger.error("Refusing stream %s: unknown call %s", event.stream_sid, event.call_sid)
        session.close()
        await socket.close(code=POLICY_VIOLATION)
        return False

    state.stream_sid = event.stream_sid
    state.call_sid = event.call_sid
    state.call = call
    state.session = session
    state.utterance = Utterance(state.settings)
    state.voice = VoiceSession(
        conversation=Conversation(session, call, build_model(state.settings), state.settings),
        stt=build_stt(state.settings),
        tts=build_tts(state.settings),
        settings=state.settings,
    )

    greeting = await anyio.to_thread.run_sync(state.voice.greeting, GREETING)
    if greeting is not None:
        await _send_audio(socket, state, greeting)
    return True


async def _media(socket: WebSocket, state: StreamState, event: MediaEvent) -> None:
    """One frame of the caller talking."""
    if not state.bound or state.utterance is None:
        # Audio before the call was identified belongs to nobody.
        return
    if event.stream_sid != state.stream_sid:
        logger.warning(
            "Ignoring a frame for stream %s on stream %s",
            event.stream_sid,
            state.stream_sid,
        )
        return
    if not event.is_inbound:
        # Our own voice coming back. Transcribing it would be a loop.
        return

    complete = state.utterance.add(event.audio)
    if complete is not None:
        await _turn(socket, state, complete)


async def _turn(socket: WebSocket, state: StreamState, mulaw: bytes) -> None:
    """One complete utterance, through the dialogue layer and back as audio."""
    assert state.voice is not None  # bound before any media is accepted

    try:
        audio = utterance_from_mulaw(mulaw)
    except TelephonyAudioError as exc:
        logger.warning("Could not read an utterance on call %s: %s", state.call_sid, exc)
        return

    try:
        # Off the event loop: this is speech recognition, a model, tool calls,
        # the database and synthesis, and the carrier keeps sending while it
        # runs. `VoiceSession` is unchanged and still entirely synchronous.
        turn: VoiceTurn = await anyio.to_thread.run_sync(state.voice.speak, audio)
    except Exception:  # noqa: BLE001 - one bad turn must not end the call
        logger.exception("A turn failed on call %s", state.call_sid)
        return

    if turn.speech is not None:
        await _send_audio(socket, state, turn.speech)


async def _send_audio(socket: WebSocket, state: StreamState, speech) -> None:
    """Play something down the line, then mark the end of it.

    Frames go out as fast as they can be written; the carrier buffers them.
    Pacing playback so it can be interrupted is barge-in, and barge-in is a
    later milestone.
    """
    try:
        mulaw = speech_to_mulaw(speech)
    except TelephonyAudioError as exc:
        logger.error("Cannot play a reply on call %s: %s", state.call_sid, exc)
        return

    for payload in frames(mulaw, FRAME_BYTES):
        await socket.send_json(media_frame(state.stream_sid, payload))

    state.marks += 1
    await socket.send_json(
        mark_frame(state.stream_sid, f"reply-{state.marks}-{uuid.uuid4().hex[:8]}")
    )
