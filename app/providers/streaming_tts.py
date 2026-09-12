"""The streaming text-to-speech boundary.

The batch interface in `tts.py` is unchanged. This is the one realtime needs:
the first audio should reach the caller long before the last of it exists.

    stream(text) → chunks(): chunk… chunk(is_final=True) → aclose()

**Chunks carry headerless PCM, not WAV.** A container per chunk would be a
header the caller cannot use and a decoder the browser cannot chain; the
format travels on the chunk instead, and whichever transport is carrying it
adds a container only if it needs one.

**This streams audio, not text.** The dialogue layer still produces one
complete reply — the model is not streamed, the tool loop is untouched, and
nothing about milestone 4 changes. What is streamed is the synthesis of that
finished sentence, which is where the waiting actually is.

Nothing here names a vendor.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.providers.speech import AudioFormat
from app.providers.tts import VoiceError, VoiceUnavailable

__all__ = [
    "AudioFormat",
    "SpeechChunk",
    "StreamingTextToSpeech",
    "VoiceError",
    "VoiceStream",
    "VoiceUnavailable",
]


@dataclass(frozen=True)
class SpeechChunk:
    """A piece of the reply, in the order it should be heard."""

    audio: bytes
    format: AudioFormat
    is_final: bool = False
    characters: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """How long this chunk plays for, at its own rate."""
        samples = len(self.audio) / 2
        return samples * 1000 / self.format.sample_rate if self.format.sample_rate else 0.0


class VoiceStream(Protocol):
    """One reply being spoken, chunk by chunk."""

    def chunks(self) -> AsyncIterator[SpeechChunk]:
        """Audio in playback order, ending with one marked final."""
        ...

    async def aclose(self) -> None:
        """Stop speaking. Called on a barge-in, and on the way out."""
        ...


class StreamingTextToSpeech(Protocol):
    """Opens synthesis streams. One per reply."""

    @property
    def format(self) -> AudioFormat:
        """What this provider emits, so a caller can check before asking."""
        ...

    def stream(self, text: str, voice: str | None = None) -> VoiceStream: ...
