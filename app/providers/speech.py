"""The audio vocabulary, and the WAV codec both speech providers need.

`AudioFormat` and `Audio` are shared by `stt.py` and `tts.py`: one description
of what a lump of audio *is*, so that speech going in and speech coming out
are talked about the same way. They live below both interfaces rather than in
either one, which is also what keeps the provider modules from importing the
audio layer above them.

Only linear PCM is described here in any depth. `encoding` is a string, not an
enum, so milestone 6 can say `"mulaw"` for Twilio without this module being
redesigned — but note that Python 3.13 removed `audioop`, so whoever writes
that conversion writes the table themselves.
"""

import io
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFormat:
    """What a lump of audio is: how it is coded, and at what rate."""

    encoding: str
    sample_rate: int
    channels: int
    container: str = "wav"

    def __str__(self) -> str:
        return (
            f"{self.encoding} {self.sample_rate}Hz "
            f"{'mono' if self.channels == 1 else f'{self.channels}ch'} "
            f"({self.container})"
        )


@dataclass(frozen=True)
class Audio:
    """Audio and what it is. Bytes alone name no sound."""

    data: bytes
    format: AudioFormat
    duration_ms: int | None = None


# Signed 16-bit little-endian PCM: two bytes a sample, which is what `wave`
# means by a sample width of 2.
PCM_S16LE = "pcm_s16le"
SAMPLE_WIDTH_BYTES = 2


class WavError(Exception):
    """The bytes are not a WAV file this system can read."""


def build_wav(pcm: bytes, audio_format: AudioFormat) -> bytes:
    """Wrap raw PCM samples in a RIFF/WAVE header.

    A provider that returns headerless PCM has told us the rate out of band;
    wrapping it here means everything downstream receives audio that describes
    itself, and a browser can play it without being told anything.
    """
    if audio_format.encoding != PCM_S16LE:
        raise WavError(f"Cannot build a WAV from {audio_format.encoding!r}.")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(audio_format.channels)
        writer.setsampwidth(SAMPLE_WIDTH_BYTES)
        writer.setframerate(audio_format.sample_rate)
        writer.writeframes(pcm)
    return buffer.getvalue()


@dataclass(frozen=True)
class WavInfo:
    """What a WAV's own header says about it."""

    format: AudioFormat
    frames: int
    duration_ms: int
    pcm: bytes


def read_wav(data: bytes) -> WavInfo:
    """Read a WAV's header and samples, or say why it could not be read."""
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            rate = reader.getframerate()
            frames = reader.getnframes()
            pcm = reader.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise WavError(f"Not a readable WAV file: {exc}") from exc

    if rate <= 0:
        raise WavError("The WAV header reports a sample rate of zero.")

    return WavInfo(
        format=AudioFormat(
            # `wave` only ever hands back uncompressed PCM, so the width is
            # what distinguishes one encoding from another here.
            encoding=PCM_S16LE if width == SAMPLE_WIDTH_BYTES else f"pcm_s{width * 8}le",
            sample_rate=rate,
            channels=channels,
            container="wav",
        ),
        frames=frames,
        duration_ms=round(frames * 1000 / rate),
        pcm=pcm,
    )
