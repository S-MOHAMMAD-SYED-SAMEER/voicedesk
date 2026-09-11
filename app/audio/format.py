"""The one audio format milestone 5 accepts, and the door it comes through.

16 kHz, mono, signed 16-bit little-endian PCM, in a WAV container — both
directions. Chosen because it needs no dependency at all (`wave` and `struct`
are standard library), because it describes itself so nothing has to be told
its sample rate out of band, and because it is the lowest common denominator
every transcriber accepts.

It is not a production telephony format. Telephony is 8 kHz µ-law, and that
arrives with the milestone that needs it — `AudioFormat.encoding` is a string
precisely so that can be said without redesigning anything. Worth knowing in
advance: Python 3.13 removed `audioop`, so the µ-law conversion will have to
be written rather than imported.

Everything entering the system passes through `validate_utterance`. Nothing
downstream re-checks, and nothing downstream has to.
"""

from app.audio.errors import AudioTooLarge, InvalidAudio
from app.providers.speech import (
    PCM_S16LE,
    SAMPLE_WIDTH_BYTES,
    Audio,
    AudioFormat,
    WavError,
    build_wav,
    read_wav,
)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_BYTES = 1_048_576


def expected_format(sample_rate: int = DEFAULT_SAMPLE_RATE) -> AudioFormat:
    """The only shape `validate_utterance` will accept."""
    return AudioFormat(
        encoding=PCM_S16LE, sample_rate=sample_rate, channels=1, container="wav"
    )


def validate_utterance(
    data: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Audio:
    """One recording from the caller, checked before anything acts on it.

    Returns `Audio` carrying the duration read from the file's own header, so
    nothing downstream has to estimate it.
    """
    if len(data) > max_bytes:
        raise AudioTooLarge(
            f"An utterance may be at most {max_bytes} bytes; this one is "
            f"{len(data)}."
        )
    if not data:
        raise InvalidAudio("The utterance is empty.")

    try:
        info = read_wav(data)
    except WavError as exc:
        raise InvalidAudio(str(exc)) from exc

    wanted = expected_format(sample_rate)
    if info.format.channels != wanted.channels:
        raise InvalidAudio(
            f"Audio must be mono; this has {info.format.channels} channels."
        )
    if info.format.sample_rate != wanted.sample_rate:
        raise InvalidAudio(
            f"Audio must be {wanted.sample_rate} Hz; this is "
            f"{info.format.sample_rate} Hz."
        )
    if info.format.encoding != PCM_S16LE:
        raise InvalidAudio(
            f"Audio must be {SAMPLE_WIDTH_BYTES * 8}-bit PCM; this is "
            f"{info.format.encoding}."
        )

    return Audio(data=data, format=wanted, duration_ms=info.duration_ms)


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_SAMPLE_RATE",
    "Audio",
    "AudioFormat",
    "build_wav",
    "expected_format",
    "read_wav",
    "validate_utterance",
]
