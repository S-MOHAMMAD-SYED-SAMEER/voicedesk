"""The speech-to-text boundary.

Everything above this line works in terms of `transcribe(audio) -> Transcript`.
Nothing above it imports a vendor SDK or knows a vendor's request shape, so
swapping one transcriber for another touches one file.

Milestone 5 is utterance-based: one complete recording in, one transcript out.
There is no chunk iterator and no partial-transcript callback, because
deciding when the caller has stopped talking is voice-activity detection, and
that belongs with barge-in in a later milestone. A streaming implementation
will add a second method here rather than replace this one.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.providers.speech import PCM_S16LE, Audio, AudioFormat

__all__ = [
    "PCM_S16LE",
    "Audio",
    "AudioFormat",
    "SpeechError",
    "SpeechToText",
    "SpeechUnavailable",
    "Transcript",
    "UnsupportedAudio",
]


@dataclass(frozen=True)
class Transcript:
    """What was heard, and how sure the provider was about it.

    `text` is `""` rather than `None` when nothing was recognised: silence is
    something that happened, and it is not the same as a provider that failed.
    The dialogue layer is never called for an empty transcript.
    """

    text: str
    # Carried, and deliberately not acted on. The specification's rule about
    # low confidence twice in a row needs endpoint detection to be meaningful,
    # so it belongs with the milestone that builds that; pretending to
    # implement it here would be worse than not having it.
    confidence: float | None = None
    language: str | None = None
    audio_ms: int | None = None
    latency_ms: int | None = None
    provider_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SpeechToText(Protocol):
    """The one call the audio layer makes to a transcriber."""

    def transcribe(self, audio: Audio) -> Transcript: ...


class SpeechError(Exception):
    """The provider could not produce a transcript.

    Raised rather than returned, and never softened into an empty transcript:
    a failure that looks like silence would have the receptionist ask the
    caller to repeat themselves while the real fault went unrecorded.
    """


class SpeechUnavailable(SpeechError):
    """The provider could not be reached, or would not serve the request."""


class UnsupportedAudio(SpeechError):
    """The provider will not accept audio in this format."""
