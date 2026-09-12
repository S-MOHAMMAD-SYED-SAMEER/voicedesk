"""The streaming speech-to-text boundary.

The batch interface in `stt.py` is unchanged and still used by the
press-to-talk paths. This is the one realtime needs: audio goes in while the
caller is still talking, and partial guesses come back before the final
answer does.

    stream(format) → send(pcm)… → finish() → events(): partial… final → aclose()

**One rule dominates the design.** A partial transcript is a guess that will
change. It may be shown, logged, or used to decide that somebody is talking.
It must never reach the dialogue layer, because a tool call made from a guess
is a booking made from a guess. Only a `FinalTranscript` crosses that line,
and there is a test whose whole job is to prove it.

Nothing here names a vendor.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.providers.speech import Audio, AudioFormat
from app.providers.stt import SpeechError, SpeechUnavailable, UnsupportedAudio

__all__ = [
    "Audio",
    "AudioFormat",
    "FinalTranscript",
    "PartialTranscript",
    "SpeechError",
    "SpeechStream",
    "SpeechUnavailable",
    "StreamingSpeechToText",
    "TranscriptEvent",
    "UnsupportedAudio",
]


@dataclass(frozen=True)
class PartialTranscript:
    """A guess, and it will change. Display and logging only."""

    text: str
    confidence: float | None = None
    at_ms: int = 0


@dataclass(frozen=True)
class FinalTranscript:
    """What the caller said. The only thing that reaches the dialogue."""

    text: str
    confidence: float | None = None
    language: str | None = None
    audio_ms: int | None = None
    provider_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


TranscriptEvent = PartialTranscript | FinalTranscript


class SpeechStream(Protocol):
    """One utterance's worth of recognition, open for as long as it lasts."""

    async def send(self, pcm: bytes) -> None:
        """Hand over audio as it arrives. Never blocks on recognition."""
        ...

    async def finish(self) -> None:
        """No more audio. Ask for whatever is left as a final."""
        ...

    def events(self) -> AsyncIterator[TranscriptEvent]:
        """Partials as they come, then exactly one final, then completion."""
        ...

    async def aclose(self) -> None:
        """Give up on this utterance and release whatever it holds."""
        ...


class StreamingSpeechToText(Protocol):
    """Opens recognition streams. One per utterance."""

    def stream(self, audio_format: AudioFormat) -> SpeechStream: ...
