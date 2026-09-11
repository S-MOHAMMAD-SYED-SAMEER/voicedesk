"""The text-to-speech boundary.

`synthesize(text) -> Speech`, and nothing else. The audio vocabulary comes
from `app.providers.speech`, shared with the transcriber, so one `AudioFormat`
describes both what the caller said and what the receptionist says back.

As with speech-to-text, milestone 5 returns one complete recording. Starting
playback on the first chunk is a latency optimisation, and the milestone that
owns latency will add a streaming method beside this one.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.providers.speech import Audio, AudioFormat

__all__ = [
    "Audio",
    "AudioFormat",
    "Speech",
    "TextToSpeech",
    "VoiceError",
    "VoiceUnavailable",
]


@dataclass(frozen=True)
class Speech:
    """Synthesised audio, and what it cost to make."""

    audio: bytes
    format: AudioFormat
    duration_ms: int | None = None
    voice: str = ""
    latency_ms: int | None = None
    provider_name: str = ""
    # The specification's cost line counts TTS characters. Recorded here so
    # the milestone that computes cost has the number; nothing in milestone 5
    # multiplies it by anything.
    characters: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TextToSpeech(Protocol):
    """The one call the audio layer makes to a synthesiser."""

    @property
    def format(self) -> AudioFormat:
        """What this provider emits, so a caller can check before asking."""
        ...

    def synthesize(self, text: str, voice: str | None = None) -> Speech: ...


class VoiceError(Exception):
    """The provider could not produce audio."""


class VoiceUnavailable(VoiceError):
    """The provider could not be reached, or would not serve the request."""
