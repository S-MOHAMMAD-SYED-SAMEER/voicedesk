"""Deciding when the caller has finished saying something.

Voice activity detection answers "is there speech in this frame?".
Endpointing answers "has the caller stopped?", which is a different and
harder question: speech is full of gaps, and treating the first of them as
the end of a sentence produces a receptionist that interrupts.

    IDLE ──speech for ≥ min_speech_ms──────────► SPEAKING   → speech started
    SPEAKING ──silence for ≥ silence_ms────────► IDLE       → speech ended
    SPEAKING ──length ≥ max_utterance_ms───────► IDLE       → speech ended

Three behaviours worth naming:

* **A pre-roll buffer.** Audio from shortly *before* the detector was
  convinced is kept and prepended, so the first syllable is not clipped. The
  previous milestone threw that audio away and lost the start of every
  sentence.
* **A minimum.** A click, a cough or a line pop is not an utterance, and
  nothing reaches a model because of one.
* **A ceiling.** A caller who never pauses is answered anyway, and the buffer
  cannot grow without limit on an open socket.

It knows nothing about carriers, sockets, providers or the dialogue. Frames
in, events out.
"""

from collections import deque
from dataclasses import dataclass

from app.audio.vad import VoiceActivityDetector


@dataclass(frozen=True)
class SpeechStarted:
    """Somebody began talking. Emitted as soon as it is clear, not at the end.

    Barge-in depends on this arriving while the receptionist is still
    speaking, which is why it is a separate event rather than a field on the
    one below.
    """

    at_ms: float


@dataclass(frozen=True)
class SpeechEnded:
    """A complete utterance, ready to be transcribed."""

    audio: bytes
    duration_ms: float
    at_ms: float
    # True when the ceiling closed it rather than the caller pausing.
    truncated: bool = False


EndpointEvent = SpeechStarted | SpeechEnded


class Endpointer:
    """Turns a continuous stream of frames into complete utterances."""

    def __init__(
        self,
        detector: VoiceActivityDetector,
        *,
        sample_rate: int = 8000,
        min_speech_ms: int = 120,
        silence_ms: int = 700,
        max_utterance_ms: int = 20_000,
        preroll_ms: int = 200,
    ) -> None:
        self._detector = detector
        self._sample_rate = sample_rate
        self._min_speech_ms = min_speech_ms
        self._silence_ms = silence_ms
        self._max_utterance_ms = max_utterance_ms
        self._preroll_ms = preroll_ms

        self._preroll: deque[bytes] = deque()
        self._preroll_held_ms = 0.0
        self._buffer = bytearray()
        self._candidate_ms = 0.0
        self._silence_held_ms = 0.0
        self._speaking = False
        self._elapsed_ms = 0.0

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms

    @property
    def utterance_ms(self) -> float:
        return self._frame_ms(bytes(self._buffer))

    def feed(self, pcm: bytes) -> list[EndpointEvent]:
        """One frame in; whatever it settled, out.

        A list because a single frame can both end one utterance and — at the
        ceiling — do so without the caller having stopped.
        """
        if not pcm:
            return []

        frame_ms = self._frame_ms(pcm)
        self._elapsed_ms += frame_ms
        speech = self._detector.analyse(pcm).speech
        events: list[EndpointEvent] = []

        if not self._speaking:
            self._remember(pcm, frame_ms)
            if speech:
                self._candidate_ms += frame_ms
                if self._candidate_ms >= self._min_speech_ms:
                    # Convinced. Everything held back comes with it.
                    self._speaking = True
                    self._silence_held_ms = 0.0
                    self._buffer = bytearray(b"".join(self._preroll))
                    self._preroll.clear()
                    self._preroll_held_ms = 0.0
                    events.append(SpeechStarted(at_ms=self._elapsed_ms))
            else:
                # Speech has to be continuous to count; a click does not add
                # up with the click a second later.
                self._candidate_ms = 0.0
            return events

        self._buffer.extend(pcm)
        self._silence_held_ms = 0.0 if speech else self._silence_held_ms + frame_ms

        if self.utterance_ms >= self._max_utterance_ms:
            events.append(self._finish(truncated=True))
        elif self._silence_held_ms >= self._silence_ms:
            events.append(self._finish(truncated=False))
        return events

    def flush(self) -> SpeechEnded | None:
        """Whatever is being said right now, because the call is ending."""
        if not self._speaking or not self._buffer:
            self.reset()
            return None
        return self._finish(truncated=True)

    def reset(self) -> None:
        """Forget the utterance in progress. Used after a barge-in."""
        self._preroll.clear()
        self._preroll_held_ms = 0.0
        self._buffer = bytearray()
        self._candidate_ms = 0.0
        self._silence_held_ms = 0.0
        self._speaking = False

    # --- internals ---------------------------------------------------------

    def _finish(self, *, truncated: bool) -> SpeechEnded:
        ended = SpeechEnded(
            audio=bytes(self._buffer),
            duration_ms=self.utterance_ms,
            at_ms=self._elapsed_ms,
            truncated=truncated,
        )
        self.reset()
        return ended

    def _remember(self, pcm: bytes, frame_ms: float) -> None:
        """Hold recent silence, in case it turns out to be the run-up."""
        self._preroll.append(pcm)
        self._preroll_held_ms += frame_ms
        while self._preroll and self._preroll_held_ms > self._preroll_ms:
            self._preroll_held_ms -= self._frame_ms(self._preroll.popleft())

    def _frame_ms(self, pcm: bytes) -> float:
        return len(pcm) / 2 * 1000 / self._sample_rate
