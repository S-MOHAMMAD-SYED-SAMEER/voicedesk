"""The realtime conversation: audio in, audio out, interruptible.

    frames ─► VAD ─► endpointer ─► streaming STT ─► Conversation ─► streaming TTS ─► sink
                       │                                                     ▲
                       └── speech during playback ── cancel, clear ──────────┘

This is the streaming counterpart to `app/audio/session.py`, which is
unchanged and still serves press-to-talk. Both call `Conversation.send`, so
there is exactly one dialogue implementation, one tool registry and one
calendar underneath them.

**The reader never waits for a turn.** `feed()` returns as soon as the frame
is accounted for; recognition, the model, the tools, the database and
synthesis all happen in a task that runs beside it. The previous milestone
awaited the turn inline, which meant no inbound audio was read while one was
running — and a receptionist that cannot hear you while it is thinking cannot
be interrupted.

**Nothing stale is ever spoken.** Every turn carries a generation number, and
audio is checked against it on the way out. This matters because cancellation
is partial by nature: `anyio.to_thread.run_sync` cannot terminate a running
thread, so a model request and the tool calls it made will finish even after
the caller has interrupted. A booking that was committed stays committed —
undoing it would be worse — but its reply is discarded and never heard, and
the dialogue is never re-run.

What this module does not know: carriers, JSON, µ-law, sockets, SQL, tools,
the calendar, or any vendor.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anyio

from app.audio.endpoint import Endpointer, SpeechEnded, SpeechStarted
from app.audio.vad import VoiceActivityDetector
from app.config import Settings, get_settings
from app.dialogue import Conversation, DialogueResult
from app.providers.speech import AudioFormat
from app.providers.streaming_stt import (
    FinalTranscript,
    PartialTranscript,
    SpeechError,
    SpeechStream,
    StreamingSpeechToText,
)
from app.providers.streaming_tts import (
    SpeechChunk,
    StreamingTextToSpeech,
    VoiceError,
)
from app.realtime.generation import Generation, Generations
from app.realtime.sink import AudioSink

logger = logging.getLogger(__name__)

# The same fixed lines the press-to-talk path uses, for the same reasons: a
# system that has just failed is not the thing to ask for an apology.
NOT_HEARD_REPLY = "Sorry, I didn't catch that — could you say that again?"
SPEECH_FAILURE_REPLY = (
    "I'm sorry, I'm having trouble hearing you. Let me pass you to a colleague."
)


@dataclass
class TurnTiming:
    """When each stage of one turn happened, in milliseconds since the epoch.

    Only what a question is actually asked of. `playback_end` lives here and
    in no database column: turn latency is derived from it, and storing a
    number the carrier may never report would be worse than deriving one.
    """

    speech_start: float | None = None
    speech_end: float | None = None
    stt_first_partial: float | None = None
    stt_final: float | None = None
    dialogue_start: float | None = None
    dialogue_end: float | None = None
    tts_first_audio: float | None = None
    playback_end: float | None = None

    @staticmethod
    def _gap(start: float | None, end: float | None) -> int | None:
        if start is None or end is None or end < start:
            return None
        return round(end - start)

    @property
    def audio_ms(self) -> int | None:
        """How long the caller spoke for."""
        return self._gap(self.speech_start, self.speech_end)

    @property
    def stt_latency_ms(self) -> int | None:
        """From the caller stopping to knowing what they said."""
        return self._gap(self.speech_end, self.stt_final)

    @property
    def tts_latency_ms(self) -> int | None:
        """From having a reply to the first audio of it existing."""
        return self._gap(self.dialogue_end, self.tts_first_audio)

    @property
    def first_audio_latency_ms(self) -> int | None:
        """The specification's number: caller stops, agent starts speaking."""
        return self._gap(self.speech_end, self.tts_first_audio)

    @property
    def turn_latency_ms(self) -> int | None:
        """From the caller stopping to the reply having been heard."""
        return self._gap(self.speech_end, self.playback_end)


@dataclass
class RealtimeTurn:
    """One exchange, and what became of it."""

    generation: int
    transcript: str = ""
    reply: str = ""
    dialogue: DialogueResult | None = None
    timing: TurnTiming = field(default_factory=TurnTiming)
    confidence: float | None = None
    interrupted: bool = False
    failed: bool = False
    # "stt" | "empty" | "dialogue" | "tts" | None
    failure: str | None = None


class RealtimeSession:
    """One call: its audio, its turns, and its one interruptible reply."""

    def __init__(
        self,
        conversation: Conversation,
        stt: StreamingSpeechToText,
        tts: StreamingTextToSpeech,
        sink: AudioSink,
        settings: Settings | None = None,
        *,
        sample_rate: int = 8000,
        on_partial: Callable[[PartialTranscript], Awaitable[None]] | None = None,
        on_turn: Callable[[RealtimeTurn], Awaitable[None]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._settings = resolved
        self._conversation = conversation
        self._stt = stt
        self._tts = tts
        self._sink = sink
        self._sample_rate = sample_rate
        self._on_partial = on_partial
        self._on_turn = on_turn
        self._clock = clock or (lambda: time.perf_counter() * 1000)

        self._endpointer = Endpointer(
            VoiceActivityDetector(
                energy_threshold=resolved.vad_energy_threshold,
                noise_floor_alpha=resolved.vad_noise_floor_alpha,
                hysteresis=resolved.vad_hysteresis,
            ),
            sample_rate=sample_rate,
            min_speech_ms=resolved.vad_min_speech_ms,
            silence_ms=resolved.endpoint_silence_ms,
            max_utterance_ms=resolved.endpoint_max_utterance_ms,
            preroll_ms=resolved.vad_preroll_ms,
        )
        self._generations = Generations()
        self._audio_format = AudioFormat(
            encoding="pcm_s16le", sample_rate=sample_rate, channels=1, container="raw"
        )
        self._playing = False
        self._voice_stream = None
        self._turn_scope: anyio.CancelScope | None = None
        self._task_group: anyio.abc.TaskGroup | None = None
        self.turns: list[RealtimeTurn] = []

    # --- lifecycle ---------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def generation(self) -> int:
        return self._generations.current

    def attach(self, task_group) -> None:
        """Give the session somewhere to run its turns.

        A task group owned by the transport, so that when the socket closes
        every turn in flight is cancelled with it rather than outliving the
        call.
        """
        self._task_group = task_group

    async def greeting(self, text: str) -> None:
        """Say the opening line.

        Not a turn: nobody prompted it, no model is asked and no transcript
        row is written. It is the receptionist picking up.
        """
        await self._speak(text, self._generations.token(), TurnTiming())

    async def feed(self, pcm: bytes) -> None:
        """One frame of the caller. Returns immediately, whatever is running."""
        for event in self._endpointer.feed(pcm):
            if isinstance(event, SpeechStarted):
                await self._on_speech_started()
            elif isinstance(event, SpeechEnded):
                await self._on_speech_ended(event)

    async def finish(self) -> None:
        """The call is ending; answer anything already said, then stop."""
        ended = self._endpointer.flush()
        if ended is not None and ended.audio:
            await self._on_speech_ended(ended)

    def playback_finished(self) -> None:
        """The transport says the caller has heard everything sent so far.

        Timing only. Whether the receptionist is still speaking is tracked by
        delivery, not by this: a transport that never reports playback would
        otherwise leave the session believing it was talking forever, and
        every later utterance would look like an interruption.
        """
        if self.turns:
            turn = self.turns[-1]
            if turn.timing.playback_end is None:
                turn.timing.playback_end = self._clock()

    # --- the two things that can happen ------------------------------------

    async def _on_speech_started(self) -> None:
        """Somebody is talking. If we were talking too, stop."""
        if not self._playing:
            return
        if not self._settings.barge_in_enabled:
            logger.debug("Speech during playback, but barge-in is disabled.")
            return

        logger.info("Barge-in: the caller interrupted.")
        if self.turns:
            self.turns[-1].interrupted = True
        await self._interrupt()

    async def _on_speech_ended(self, ended: SpeechEnded) -> None:
        """A complete utterance. Start a turn beside the reader, not in it."""
        generation = self._generations.token(self._generations.next())
        timing = TurnTiming(
            speech_start=self._clock() - ended.duration_ms, speech_end=self._clock()
        )
        turn = RealtimeTurn(generation=generation.number, timing=timing)
        self.turns.append(turn)

        if self._task_group is None:
            # No task group: run inline. Only tests do this, and only when
            # they are asserting on the turn rather than on concurrency.
            await self._run_turn(ended.audio, generation, turn)
            return
        self._task_group.start_soon(self._run_turn, ended.audio, generation, turn)

    # --- one turn ----------------------------------------------------------

    async def _run_turn(
        self, audio: bytes, generation: Generation, turn: RealtimeTurn
    ) -> None:
        with anyio.CancelScope() as scope:
            self._turn_scope = scope
            try:
                await self._transcribe_and_answer(audio, generation, turn)
            except Exception:  # noqa: BLE001 - one turn must not end the call
                logger.exception("A realtime turn failed.")
                turn.failed, turn.failure = True, turn.failure or "dialogue"
            finally:
                if self._turn_scope is scope:
                    self._turn_scope = None
                if self._on_turn is not None:
                    # Awaited, not queued: a transport that wants to report this
                    # turn should be able to do it now, rather than when the
                    # next inbound frame happens to arrive.
                    await self._on_turn(turn)

    async def _transcribe_and_answer(
        self, audio: bytes, generation: Generation, turn: RealtimeTurn
    ) -> None:
        try:
            final = await self._recognise(audio, generation, turn)
        except SpeechError as exc:
            # A broken provider must never look like silence, and no
            # transcript is invented for it.
            logger.warning("Recognition failed: %s", exc)
            turn.failed, turn.failure = True, "stt"
            turn.reply = SPEECH_FAILURE_REPLY
            await self._speak(SPEECH_FAILURE_REPLY, generation, turn.timing)
            return

        if generation.stale:
            logger.info("Discarding a transcript from an interrupted turn.")
            return
        if final is None or not final.text.strip():
            # Silence is not a question. The model is not asked about it.
            turn.failed, turn.failure = True, "empty"
            turn.reply = NOT_HEARD_REPLY
            await self._speak(NOT_HEARD_REPLY, generation, turn.timing)
            return

        turn.transcript = final.text
        turn.confidence = final.confidence

        turn.timing.dialogue_start = self._clock()
        # Off the event loop: a model request, tool calls and the database,
        # while the caller keeps talking. `Conversation` is unchanged and
        # still entirely synchronous.
        result = await anyio.to_thread.run_sync(
            self._conversation.send, final.text, abandon_on_cancel=True
        )
        turn.timing.dialogue_end = self._clock()
        turn.dialogue = result
        turn.reply = result.text
        if result.failed:
            turn.failed, turn.failure = True, "dialogue"

        if generation.stale:
            # The caller interrupted while this was in flight. Whatever the
            # tools did stands — it is already committed, and undoing it would
            # be worse — but nobody hears a reply to a question they abandoned.
            logger.info("Discarding a reply from an interrupted turn.")
            return

        await self._speak(result.text, generation, turn.timing)

    async def _recognise(
        self, audio: bytes, generation: Generation, turn: RealtimeTurn
    ) -> FinalTranscript | None:
        """Feed one utterance to the transcriber and wait for its final."""
        stream: SpeechStream = self._stt.stream(self._audio_format)
        final: FinalTranscript | None = None
        try:
            await stream.send(audio)
            await stream.finish()
            async for event in stream.events():
                if generation.stale:
                    break
                if isinstance(event, PartialTranscript):
                    if turn.timing.stt_first_partial is None:
                        turn.timing.stt_first_partial = self._clock()
                    # Display and logging only. A guess must never reach the
                    # dialogue layer, because a tool called from a guess is a
                    # booking made from a guess.
                    if self._on_partial is not None:
                        await self._on_partial(event)
                    continue
                if final is None:
                    final = event
                    turn.timing.stt_final = self._clock()
                else:
                    logger.info("Ignoring a second final transcript.")
        finally:
            await stream.aclose()
        return final

    # --- speaking ----------------------------------------------------------

    async def _speak(
        self, text: str, generation: Generation, timing: TurnTiming
    ) -> None:
        """Synthesise and play, checking on every chunk that it is still wanted."""
        if generation.stale or not text:
            return

        stream = self._tts.stream(text)
        self._voice_stream = stream
        self._playing = True
        try:
            async for chunk in stream.chunks():
                if generation.stale:
                    # Providers have chunks in flight when they are closed.
                    # This is the check that keeps them off the line.
                    logger.info("Dropping audio from an interrupted turn.")
                    return
                if timing.tts_first_audio is None:
                    timing.tts_first_audio = self._clock()
                await self._sink.send(chunk)
        except VoiceError as exc:
            # The dialogue already happened and is already persisted. It is
            # not run again: retrying for audio would risk acting twice.
            logger.error("Synthesis failed: %s", exc)
            self._playing = False
            return
        finally:
            if self._voice_stream is stream:
                self._voice_stream = None
                # Delivery is over. On a paced transport that is the whole
                # length of the reply, which is exactly the window in which
                # an interruption means something.
                self._playing = False
            await stream.aclose()

        if generation.current:
            await self._sink.mark(f"reply-{generation.number}")

    async def _interrupt(self) -> None:
        """Make everything in flight stale, silence it, and empty the line."""
        self._generations.next()

        stream = self._voice_stream
        self._voice_stream = None
        if stream is not None:
            await stream.aclose()

        scope = self._turn_scope
        if scope is not None:
            scope.cancel()

        # Order matters: the generation is already bumped, so nothing new can
        # be queued between closing the stream and emptying what is queued.
        await self._sink.clear()
        self._playing = False
