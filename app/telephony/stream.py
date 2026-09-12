"""The media stream: a telephone on one end, the same receptionist on the other.

    carrier ──► WS /telephony/stream ──► µ-law → PCM ──► RealtimeSession
                                                              │
                                    Conversation → tools → calendar → PostgreSQL
                                                              │
    carrier ◄── media frames ◄── µ-law ◄── streaming TTS ◄─────┘

There is no second dialogue here. The realtime session makes the same
`Conversation.send` call the browser harness makes; only the audio at either
end is different, and that conversion lives in `app/audio/telephony.py`.

**The reader never waits for a turn.** Frames are read in one task and turns
run in another, so the caller can be heard while the receptionist is thinking
— which is the whole of what makes an interruption possible. The previous
milestone awaited each turn inline and was deaf for its duration.

**Replies are paced.** Audio is written roughly in real time rather than as
fast as the socket accepts it, because a carrier buffers whatever it is given:
sending a whole reply at once means that by the time somebody interrupts, they
have already heard it. Pacing is what leaves something for `clear` to discard.

With `realtime_enabled` off — the default — none of the above happens and the
milestone-6 path runs instead: one complete utterance, then one complete
reply. Both are here, and they share everything below `Conversation`.
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
    FRAME_MS,
    TELEPHONY_SAMPLE_RATE,
    TelephonyAudioError,
    downsample_16k_to_8k,
    frames,
    mean_amplitude,
    mulaw_decode,
    mulaw_encode,
    speech_to_mulaw,
    utterance_from_mulaw,
)
from app.config import Settings, get_settings
from app.cost import record_call_cost, record_turn_cost
from app.db.session import get_sessionmaker
from app.dialogue import Conversation
from app.models import Call, CallDirection
from app.providers.factory import (
    build_model,
    build_streaming_stt,
    build_streaming_tts,
    build_stt,
    build_tts,
    model_provider_name,
)
from app.providers.streaming_tts import SpeechChunk
from app.realtime import RealtimeSession, RealtimeTurn
from app.telephony.events import (
    ConnectedEvent,
    MalformedEvent,
    MarkEvent,
    MediaEvent,
    StartEvent,
    StopEvent,
    UnknownEvent,
    UnsupportedMediaFormat,
    clear_frame,
    media_frame,
    mark_frame,
    parse_event,
    require_supported_format,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telephony"])

GREETING = "Thanks for calling. How can I help?"
POLICY_VIOLATION = 1008
# Who carried the call, for the accounting row. The name of the carrier, not
# of its SDK: nothing here imports one.
CARRIER_NAME = "twilio"
BYTES_PER_MS = TELEPHONY_SAMPLE_RATE // 1000


class Utterance:
    """Continuous audio, cut into turns by an amplitude timer.

    The milestone-6 boundary, kept for the path that still uses it. It is not
    voice-activity detection; `app/audio/vad.py` is what the realtime path
    uses instead.
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
            return None

        self._buffer.extend(mulaw)

        if len(self._buffer) >= self._max_bytes or self.duration_ms >= self._max_ms:
            return self.flush()
        if self._silence_ms >= self._silence_limit_ms:
            return self.flush()
        return None

    def flush(self) -> bytes | None:
        audio = bytes(self._buffer) if self._buffer else None
        self._buffer = bytearray()
        self._silence_ms = 0.0
        self._speaking = False
        return audio


class TwilioSink:
    """Where a reply goes on a telephone line.

    Paced, so an interruption has something to interrupt, and generation-aware
    through the session that owns it: nothing reaches the carrier once the
    turn that produced it is stale.
    """

    def __init__(
        self, socket: WebSocket, stream_sid: str, *, pace: bool = True
    ) -> None:
        self._socket = socket
        self._stream_sid = stream_sid
        self._pace = pace
        self.sent_frames = 0

    async def send(self, chunk: SpeechChunk) -> None:
        if not chunk.audio:
            return
        mulaw = _to_mulaw(chunk)
        for payload in frames(mulaw, FRAME_BYTES):
            await self._socket.send_json(media_frame(self._stream_sid, payload))
            self.sent_frames += 1
            if self._pace:
                # Roughly real time. Without this the carrier holds the whole
                # reply and `clear` arrives after the caller has heard it.
                await anyio.sleep(len(payload) / BYTES_PER_MS / 1000)

    async def clear(self) -> None:
        await self._socket.send_json(clear_frame(self._stream_sid))

    async def mark(self, name: str) -> None:
        await self._socket.send_json(mark_frame(self._stream_sid, name))


def _to_mulaw(chunk: SpeechChunk) -> bytes:
    """A chunk of synthesised PCM, as the carrier wants it."""
    if chunk.format.encoding != "pcm_s16le" or chunk.format.channels != 1:
        raise TelephonyAudioError(f"Cannot put {chunk.format} on a telephone line.")
    if chunk.format.sample_rate == TELEPHONY_SAMPLE_RATE:
        return mulaw_encode(chunk.audio)
    if chunk.format.sample_rate == 2 * TELEPHONY_SAMPLE_RATE:
        return mulaw_encode(downsample_16k_to_8k(chunk.audio))
    raise TelephonyAudioError(
        f"Cannot convert {chunk.format.sample_rate} Hz to "
        f"{TELEPHONY_SAMPLE_RATE} Hz."
    )


@dataclass
class StreamState:
    """One call's worth of everything, shared with no other call."""

    settings: Settings
    stream_sid: str = ""
    call_sid: str = ""
    call: Call | None = None
    session: Session | None = None
    voice: VoiceSession | None = None
    realtime: RealtimeSession | None = None
    utterance: Utterance | None = None
    marks: int = field(default=0)

    @property
    def bound(self) -> bool:
        return self.call is not None

    def close(self) -> None:
        """Record that the call ended, and let the database connection go.

        `ended_at`, and then how long the line was open. `outcome` stays null
        — summarising how a call went belongs to the milestone that owns
        post-call reporting. `total_cost_usd` is left to the cost recorder,
        which will only fill it in if every component of the call was priced;
        the line never is, so on a telephone call it stays null by design.
        """
        if self.session is not None:
            try:
                if self.call is not None:
                    self.call.ended_at = datetime.now(UTC)
                    self.session.commit()
                    # Measured duration, recorded unpriced. What a carrier
                    # bills is its own record of the call, rounded up, which
                    # this process never sees.
                    record_call_cost(
                        self.session, self.call, CARRIER_NAME, self.settings
                    )
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
        async with anyio.create_task_group() as turns:
            # Turns run beside the reader, not inside it. When this scope
            # exits — a stop, a disconnect, an error — everything in flight is
            # cancelled with the call rather than outliving it.
            await _converse(socket, state, turns)
            turns.cancel_scope.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        state.close()


async def _converse(socket: WebSocket, state: StreamState, turns) -> None:
    """Read frames until the call ends.

    One malformed frame is logged and stepped over. A carrier is entitled to
    send an event this milestone has never heard of.
    """
    while True:
        message = await socket.receive()
        if message.get("type") == "websocket.disconnect":
            return

        raw = message.get("text")
        if raw is None:
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
            if not await _start(socket, state, event, turns):
                return
            continue
        if isinstance(event, StopEvent):
            await _stopping(state)
            return
        if isinstance(event, MarkEvent):
            _played(state, event)
            continue
        if isinstance(event, MediaEvent):
            await _media(socket, state, event)


async def _start(
    socket: WebSocket, state: StreamState, event: StartEvent, turns
) -> bool:
    """Identify the call, build everything it needs, and say hello."""
    if state.bound:
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
        logger.error(
            "Refusing stream %s: unknown call %s", event.stream_sid, event.call_sid
        )
        session.close()
        await socket.close(code=POLICY_VIOLATION)
        return False

    state.stream_sid = event.stream_sid
    state.call_sid = event.call_sid
    state.call = call
    state.session = session
    conversation = Conversation(
        session, call, build_model(state.settings), state.settings
    )

    if state.settings.realtime_enabled:
        state.realtime = RealtimeSession(
            conversation=conversation,
            stt=build_streaming_stt(state.settings),
            tts=build_streaming_tts(state.settings),
            sink=TwilioSink(socket, event.stream_sid),
            settings=state.settings,
            sample_rate=TELEPHONY_SAMPLE_RATE,
            on_turn=lambda turn: _record_latency(state, turn),
        )
        state.realtime.attach(turns)
        await state.realtime.greeting(GREETING)
        return True

    state.utterance = Utterance(state.settings)
    state.voice = VoiceSession(
        conversation=conversation,
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
    if not state.bound:
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

    if state.realtime is not None:
        # Returns as soon as the frame is accounted for; whatever it started
        # runs beside this loop.
        await state.realtime.feed(mulaw_decode(event.audio))
        return

    if state.utterance is None:
        return
    complete = state.utterance.add(event.audio)
    if complete is not None:
        await _turn(socket, state, complete)


def _played(state: StreamState, event: MarkEvent) -> None:
    """The carrier says the audio up to this marker has been heard."""
    if state.realtime is not None and event.stream_sid == state.stream_sid:
        state.realtime.playback_finished()


async def _stopping(state: StreamState) -> None:
    """The caller hung up. Answer anything already said, then stop."""
    if state.realtime is not None:
        with anyio.move_on_after(0.1):
            await state.realtime.finish()


async def _record_latency(state: StreamState, turn: RealtimeTurn) -> None:
    """Write what only the audio layer could measure onto the turn rows.

    The dialogue layer wrote those rows and returned their ids; it cannot know
    how long the caller spoke or how long recognition took. Best effort: a
    call is not worth failing over a metric.

    What the turn consumed is written here too, by the one module allowed to
    write a cost row. Off unless `cost_tracking_enabled` says otherwise.
    """
    result = turn.dialogue
    if result is None or state.session is None:
        return
    from app.realtime.latency import record

    record(state.session, result, turn.timing)
    _record_cost(state, turn)


def _record_cost(state: StreamState, turn) -> None:
    """What one turn consumed, for whichever of the two paths produced it."""
    if state.session is None or turn.dialogue is None:
        return
    record_turn_cost(
        state.session,
        turn.dialogue,
        turn,
        state.settings,
        llm_provider=model_provider_name(state.settings),
    )


async def _turn(socket: WebSocket, state: StreamState, mulaw: bytes) -> None:
    """One complete utterance, the milestone-6 way."""
    assert state.voice is not None

    try:
        audio = utterance_from_mulaw(mulaw)
    except TelephonyAudioError as exc:
        logger.warning(
            "Could not read an utterance on call %s: %s", state.call_sid, exc
        )
        return

    try:
        turn: VoiceTurn = await anyio.to_thread.run_sync(state.voice.speak, audio)
    except Exception:  # noqa: BLE001 - one bad turn must not end the call
        logger.exception("A turn failed on call %s", state.call_sid)
        return

    _record_cost(state, turn)

    if turn.speech is not None:
        await _send_audio(socket, state, turn.speech)


async def _send_audio(socket: WebSocket, state: StreamState, speech) -> None:
    """Play a complete reply down the line, then mark the end of it."""
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
