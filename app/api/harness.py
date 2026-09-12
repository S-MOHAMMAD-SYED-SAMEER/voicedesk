"""The browser harness: a microphone on one end, the receptionist on the other.

    push-to-talk:  browser mic → one WAV → VoiceSession → JSON + WAV
    realtime:      browser mic → 20 ms PCM frames → RealtimeSession → JSON + PCM

It exists so the dialogue can be developed against a real voice without
spending phone credits, which is exactly what the specification asks for.
It is not a product surface and it is not telephony: Twilio, media streams
and real phone numbers arrive with the next milestone.

Why a WebSocket rather than a POST, given that milestone 5 sends whole
utterances: a `Conversation` holds its history in memory, so the connection
*is* the call. Opening one starts a call, closing one ends it, and nothing
needs a session registry keyed by an identifier the browser could get wrong.

Both modes live here and share everything below `Conversation`. Push-to-talk
is unchanged and remains the default; realtime is what `realtime_enabled`
switches on, and a `mode` message flips between them mid-connection so the
two can be compared without restarting anything. In realtime a binary frame
from the browser is 20 ms of raw PCM and a binary frame from the server is a
chunk of the reply; in push-to-talk both are complete WAVs.

What this module owns: the socket, the call's lifetime, frame decoding, size
and format checks, and the order replies go out in. What it does not own:
anything about dialogue, tools, the calendar, or a vendor's API.
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

import anyio

from app.audio import AudioError, VoiceSession, VoiceTurn, utterance_from_bytes
from app.config import Settings, get_settings
from app.db.session import get_sessionmaker
from app.dialogue import Conversation
from app.models import Call, CallDirection
from app.providers.factory import (
    build_model,
    build_streaming_stt,
    build_streaming_tts,
    build_stt,
    build_tts,
)
from app.providers.streaming_stt import PartialTranscript
from app.providers.streaming_tts import SpeechChunk
from app.realtime import RealtimeSession, RealtimeTurn, record_latency
from app.static import harness_page

logger = logging.getLogger(__name__)

router = APIRouter(tags=["harness"])

# `calls.from_number` and `to_number` are NOT NULL, and a browser has neither.
# These sentinels say plainly that the call came from the development harness.
# They are deliberately not telephone-shaped: a fake number would eventually
# be read as a real one, and making the columns nullable would be a migration
# to accommodate a test tool.
BROWSER_FROM_NUMBER = "browser"
BROWSER_TO_NUMBER = "harness"

GREETING = "Thanks for calling. How can I help?"


@router.get("/harness", response_class=HTMLResponse)
def harness() -> HTMLResponse:
    """The development page. One button, one log, no build step."""
    return HTMLResponse(harness_page())


class BrowserSink:
    """Where a reply goes in the browser: binary chunks, in playback order."""

    def __init__(self, socket: WebSocket) -> None:
        self._socket = socket
        self.sent = 0

    async def send(self, chunk: SpeechChunk) -> None:
        if not chunk.audio:
            return
        self.sent += 1
        await self._socket.send_bytes(chunk.audio)

    async def clear(self) -> None:
        await self._socket.send_json({"type": "clear"})

    async def mark(self, name: str) -> None:
        await self._socket.send_json({"type": "mark", "name": name})


@router.websocket("/ws/harness")
async def harness_socket(socket: WebSocket) -> None:
    """One browser call, from answering to hanging up."""
    settings = get_settings()
    await socket.accept()

    with get_sessionmaker()() as session:
        call = _start_call(session)
        conversation = Conversation(session, call, build_model(settings), settings)
        voice = VoiceSession(
            conversation=conversation,
            stt=build_stt(settings),
            tts=build_tts(settings),
            settings=settings,
        )

        try:
            async with anyio.create_task_group() as turns:
                realtime = _build_realtime(socket, session, conversation, settings)
                realtime.attach(turns)

                greeting = voice.greeting(GREETING)
                await socket.send_json(
                    {
                        "type": "ready",
                        "call_id": str(call.id),
                        "greeting": GREETING,
                        "realtime": settings.realtime_enabled,
                        "format": {
                            "encoding": "pcm_s16le",
                            "sample_rate": settings.audio_sample_rate,
                            "channels": 1,
                            "container": "wav",
                        },
                        "frame_ms": settings.audio_frame_ms,
                        "max_utterance_bytes": settings.max_utterance_bytes,
                        "audio_bytes": len(greeting.audio) if greeting else 0,
                    }
                )
                if greeting is not None:
                    await socket.send_bytes(greeting.audio)

                await _converse(
                    socket, voice, realtime, settings, str(call.id), call
                )
                turns.cancel_scope.cancel()
        except WebSocketDisconnect:
            pass
        finally:
            _end_call(session, call)


def _build_realtime(
    socket: WebSocket, session, conversation: Conversation, settings: Settings
) -> RealtimeSession:
    """The streaming session, built whether or not it ends up being used.

    Cheap to construct and nothing happens until a frame is fed to it, so the
    `mode` message can switch modes without rebuilding anything mid-call.
    """

    async def on_partial(event: PartialTranscript) -> None:
        # A guess, shown and nothing more. It never reaches the dialogue.
        await socket.send_json({"type": "partial", "text": event.text})

    async def on_turn(turn: RealtimeTurn) -> None:
        if turn.dialogue is not None:
            record_latency(session, turn.dialogue, turn.timing)
        await socket.send_json(_realtime_message(turn))

    return RealtimeSession(
        conversation=conversation,
        stt=build_streaming_stt(settings),
        tts=build_streaming_tts(settings),
        sink=BrowserSink(socket),
        settings=settings,
        sample_rate=settings.audio_sample_rate,
        on_partial=on_partial,
        on_turn=on_turn,
    )


async def _converse(
    socket: WebSocket,
    voice: VoiceSession,
    realtime: RealtimeSession,
    settings: Settings,
    voice_call_id: str,
    call: Call,
) -> None:
    """Read frames until the caller hangs up or the socket closes."""
    streaming = settings.realtime_enabled

    while True:
        frame = await socket.receive()

        if frame.get("type") == "websocket.disconnect":
            return

        text = frame.get("text")
        if text is not None:
            if '"hangup"' in text or text.strip() == "hangup":
                if streaming:
                    await realtime.finish()
                return
            if '"mode"' in text:
                # Flip between the two paths mid-call, so they can be
                # compared without restarting anything.
                streaming = '"realtime"' in text
                await socket.send_json({"type": "mode", "realtime": streaming})
                continue
            await _error(socket, "unexpected_text_frame")
            continue

        data = frame.get("bytes")
        if data is None:
            await _error(socket, "empty_frame")
            continue

        if streaming:
            # 20 ms of raw PCM. Returns as soon as the frame is accounted
            # for; whatever it started runs beside this loop and reports
            # itself when it is done.
            await realtime.feed(data)
            continue

        try:
            audio = utterance_from_bytes(data, settings)
        except AudioError as exc:
            # Rejected before any provider is asked to make sense of it.
            await _error(socket, "invalid_audio", str(exc))
            continue

        try:
            turn = await anyio.to_thread.run_sync(voice.speak, audio)
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the call
            # `VoiceSession` converts every failure it anticipates into a
            # `VoiceTurn`. Anything reaching here is unanticipated — a
            # misconfigured provider, say, whose SDK raises a plain TypeError
            # when it cannot find a credential. The harness says so and keeps
            # the socket open; dropping the connection would look to whoever
            # is testing like a bug in the browser.
            logger.exception("Turn failed on call %s", voice_call_id)
            await _error(socket, "turn_failed", f"{type(exc).__name__}: {exc}")
            continue

        # JSON first, then the audio it describes, always in that order, so
        # the browser never has to guess what a binary frame belongs to.
        await socket.send_json(_turn_message(turn))
        if turn.speech is not None:
            await socket.send_bytes(turn.speech.audio)


def _realtime_message(turn: RealtimeTurn) -> dict[str, object]:
    return {
        "type": "turn",
        "transcript": turn.transcript,
        "confidence": turn.confidence,
        "reply": turn.reply,
        "failed": turn.failed,
        "failure": turn.failure,
        "interrupted": turn.interrupted,
        "audio_bytes": 0,
        "timings": {
            "stt_ms": turn.timing.stt_latency_ms or 0,
            "llm_ms": turn.dialogue.llm_latency_ms if turn.dialogue else 0,
            "tts_ms": turn.timing.tts_latency_ms or 0,
            "total_ms": turn.timing.first_audio_latency_ms or 0,
        },
        "tool_calls": [
            {"name": record.tool_name, "success": record.success}
            for record in (turn.dialogue.tool_calls if turn.dialogue else [])
        ],
    }


def _turn_message(turn: VoiceTurn) -> dict[str, object]:
    return {
        "type": "turn",
        "transcript": turn.transcript,
        "confidence": turn.confidence,
        "reply": turn.reply,
        "failed": turn.failed,
        "failure": turn.failure,
        "audio_bytes": len(turn.speech.audio) if turn.speech else 0,
        # Reported to the page, stored nowhere. Durable latency belongs to the
        # milestone that owns observability.
        "timings": {
            "stt_ms": turn.stt_latency_ms,
            "llm_ms": turn.dialogue.llm_latency_ms if turn.dialogue else 0,
            "tts_ms": turn.tts_latency_ms,
            "total_ms": turn.total_latency_ms,
        },
        "tool_calls": [
            {"name": record.tool_name, "success": record.success}
            for record in (turn.dialogue.tool_calls if turn.dialogue else [])
        ],
    }


async def _error(socket: WebSocket, reason: str, detail: str = "") -> None:
    await socket.send_json({"type": "error", "reason": reason, "detail": detail})


def _start_call(session: Session) -> Call:
    call = Call(
        direction=CallDirection.INBOUND,
        from_number=BROWSER_FROM_NUMBER,
        to_number=BROWSER_TO_NUMBER,
    )
    session.add(call)
    session.commit()
    return call


def _end_call(session: Session, call: Call) -> None:
    """Mark the call over.

    Only `ended_at`. `calls.outcome` stays null: summarising how a call went
    belongs to the milestone that owns post-call reporting, and a guess
    written now would be indistinguishable from a fact later. If this process
    dies before getting here, `ended_at` stays null too — nobody hung up.
    """
    try:
        call.ended_at = datetime.now(UTC)
        session.commit()
    except Exception:  # pragma: no cover - cleanup must not mask the real error
        logger.exception("Could not record the end of call %s", call.id)
        session.rollback()
