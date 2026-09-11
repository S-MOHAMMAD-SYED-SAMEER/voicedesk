"""Telephone audio: G.711 µ-law at 8 kHz, and the internal format either side.

Telephony speaks µ-law: 8 kHz, mono, one byte a sample, logarithmically
companded. Everything inside VoiceDesk speaks 16 kHz mono signed 16-bit PCM in
a WAV container. This module is the whole of the conversion between them, and
it knows nothing about who is carrying the audio — no JSON, no protocol, no
vendor. That lives in `app/telephony/`.

    inbound   µ-law 8 kHz ──decode──► PCM 8 kHz ──×2──► PCM 16 kHz ──► WAV
    outbound  WAV 16 kHz ──read──► PCM 16 kHz ──÷2──► PCM 8 kHz ──encode──► µ-law

**Written, not imported.** Python 3.13 removed `audioop` (PEP 594), and the
alternatives — a backport, NumPy, ffmpeg — are all out of proportion to what
this is: a 256-value codec and a 2:1 rate change. Both are pure standard
library, exact, and completely testable.

**The resampler is deliberately simple and makes nothing sound better.**
Upsampling interpolates between neighbours and downsampling averages pairs.
That is a crude anti-alias filter, and telephone audio is band-limited to
about 3.4 kHz whatever is done to it — going to 16 kHz invents no detail that
8 kHz did not carry. No claim is made here about transcription accuracy on
phone audio, and none should be made anywhere else either.
"""

import struct

from app.providers.speech import (
    PCM_S16LE,
    Audio,
    AudioFormat,
    build_wav,
    read_wav,
)
from app.providers.tts import Speech

# G.711 µ-law, as in the standard's reference implementation.
BIAS = 0x84
CLIP = 32635
MULAW = "mulaw"
TELEPHONY_SAMPLE_RATE = 8000
INTERNAL_SAMPLE_RATE = 16000
# 8000 samples a second, one byte each: 160 bytes is 20 ms, which is the frame
# size telephony carriers work in.
FRAME_BYTES = 160
FRAME_MS = 20


class TelephonyAudioError(Exception):
    """Audio that cannot be converted to or from telephone format."""


def telephony_format() -> AudioFormat:
    """What a carrier sends and expects: 8 kHz mono µ-law, no container."""
    return AudioFormat(
        encoding=MULAW,
        sample_rate=TELEPHONY_SAMPLE_RATE,
        channels=1,
        container="raw",
    )


# --- the codec -------------------------------------------------------------


def _encode_sample(sample: int) -> int:
    """One signed 16-bit sample as one µ-law byte."""
    sign = 0x80 if sample < 0 else 0
    magnitude = -sample if sample < 0 else sample
    magnitude = min(magnitude, CLIP) + BIAS

    exponent = 7
    mask = 0x4000
    while exponent > 0 and not magnitude & mask:
        exponent -= 1
        mask >>= 1
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def _decode_byte(code: int) -> int:
    """One µ-law byte as one signed 16-bit sample."""
    code = ~code & 0xFF
    magnitude = (((code & 0x0F) << 3) + BIAS) << ((code >> 4) & 0x07)
    magnitude -= BIAS
    return -magnitude if code & 0x80 else magnitude


# Built once at import: 256 entries out, 65536 in. Tables rather than
# arithmetic per sample, because a thirty-second utterance is a quarter of a
# million samples and this runs while somebody is holding a telephone.
_DECODE_TABLE = tuple(_decode_byte(code) for code in range(256))
_ENCODE_TABLE = bytes(_encode_sample(value - 32768) for value in range(65536))


def mulaw_decode(data: bytes) -> bytes:
    """µ-law bytes to signed 16-bit little-endian PCM."""
    if not data:
        return b""
    samples = [_DECODE_TABLE[byte] for byte in data]
    return struct.pack(f"<{len(samples)}h", *samples)


def mulaw_encode(pcm: bytes) -> bytes:
    """Signed 16-bit little-endian PCM to µ-law bytes.

    A trailing odd byte cannot be half a sample, so it is refused rather than
    quietly dropped — a length that is not a whole number of samples means
    something upstream is wrong.
    """
    if not pcm:
        return b""
    if len(pcm) % 2:
        raise TelephonyAudioError(
            f"PCM must be a whole number of 16-bit samples; got {len(pcm)} bytes."
        )
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return bytes(_ENCODE_TABLE[sample + 32768] for sample in samples)


# --- rate conversion -------------------------------------------------------


def upsample_8k_to_16k(pcm: bytes) -> bytes:
    """8 kHz to 16 kHz: each sample, then the midpoint to its neighbour.

    The last sample has no neighbour, so it is repeated. Linear interpolation
    adds no information — it only avoids the stair-stepping that plain sample
    duplication produces.
    """
    if not pcm:
        return b""
    if len(pcm) % 2:
        raise TelephonyAudioError(
            f"PCM must be a whole number of 16-bit samples; got {len(pcm)} bytes."
        )

    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    out: list[int] = []
    last = len(samples) - 1
    for index, sample in enumerate(samples):
        following = samples[index + 1] if index < last else sample
        out.append(sample)
        out.append((sample + following) // 2)
    return struct.pack(f"<{len(out)}h", *out)


def downsample_16k_to_8k(pcm: bytes) -> bytes:
    """16 kHz to 8 kHz: the mean of each pair.

    An odd number of samples leaves one over, which is dropped: half a pair is
    not a sample, and at 16 kHz it is 62 microseconds.
    """
    if not pcm:
        return b""
    if len(pcm) % 2:
        raise TelephonyAudioError(
            f"PCM must be a whole number of 16-bit samples; got {len(pcm)} bytes."
        )

    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    out = [
        (samples[index] + samples[index + 1]) // 2
        for index in range(0, len(samples) - 1, 2)
    ]
    return struct.pack(f"<{len(out)}h", *out)


# --- the two conversions the telephony layer actually calls -----------------


def utterance_from_mulaw(mulaw: bytes) -> Audio:
    """One complete utterance off the wire, as the audio layer expects it.

    The result is exactly what `validate_utterance` accepts and what
    `VoiceSession.speak` is given by the browser — which is the point. There is
    one dialogue path, and this is how a telephone joins it.
    """
    if not mulaw:
        raise TelephonyAudioError("There is no audio in this utterance.")

    pcm = upsample_8k_to_16k(mulaw_decode(mulaw))
    internal = AudioFormat(
        encoding=PCM_S16LE,
        sample_rate=INTERNAL_SAMPLE_RATE,
        channels=1,
        container="wav",
    )
    return Audio(
        data=build_wav(pcm, internal),
        format=internal,
        duration_ms=round(len(mulaw) * 1000 / TELEPHONY_SAMPLE_RATE),
    )


def speech_to_mulaw(speech: Speech) -> bytes:
    """A synthesised reply, ready to go down a telephone line.

    Only the rates a carrier can be reached at are accepted: 16 kHz is halved,
    8 kHz passes through, and anything else is refused rather than resampled by
    a ratio this module does not implement.
    """
    info = read_wav(speech.audio)
    if info.format.channels != 1:
        raise TelephonyAudioError(
            f"Telephone audio is mono; this has {info.format.channels} channels."
        )
    if info.format.encoding != PCM_S16LE:
        raise TelephonyAudioError(
            f"Expected 16-bit PCM to convert; got {info.format.encoding}."
        )

    if info.format.sample_rate == INTERNAL_SAMPLE_RATE:
        pcm = downsample_16k_to_8k(info.pcm)
    elif info.format.sample_rate == TELEPHONY_SAMPLE_RATE:
        pcm = info.pcm
    else:
        raise TelephonyAudioError(
            f"Cannot convert {info.format.sample_rate} Hz to "
            f"{TELEPHONY_SAMPLE_RATE} Hz; only {INTERNAL_SAMPLE_RATE} Hz and "
            f"{TELEPHONY_SAMPLE_RATE} Hz are handled."
        )
    return mulaw_encode(pcm)


def frames(mulaw: bytes, size: int = FRAME_BYTES) -> list[bytes]:
    """Split µ-law audio into carrier-sized frames.

    A short final frame is kept rather than padded: silence added to the end of
    a reply is silence the caller waits through.
    """
    return [mulaw[start : start + size] for start in range(0, len(mulaw), size)]


def mean_amplitude(pcm: bytes) -> float:
    """Mean absolute sample value, for the milestone-6 silence boundary.

    Loudness, crudely. It is not a speech detector and is not used as one.
    """
    if len(pcm) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return sum(abs(sample) for sample in samples) / len(samples)
