"""The one audio format milestone 5 accepts, and everything it refuses."""

import pytest

from app.audio import AudioTooLarge, InvalidAudio, expected_format, validate_utterance
from app.providers.speech import (
    PCM_S16LE,
    AudioFormat,
    WavError,
    build_wav,
    read_wav,
)

from .conftest import wav_bytes


# --- the codec ------------------------------------------------------------


def test_pcm_round_trips_through_a_wav_container() -> None:
    fmt = expected_format()
    pcm = b"\x01\x02" * 8000

    info = read_wav(build_wav(pcm, fmt))

    assert info.pcm == pcm
    assert info.format == fmt


def test_duration_is_read_from_the_header_not_estimated() -> None:
    assert read_wav(wav_bytes(1500)).duration_ms == 1500
    assert read_wav(wav_bytes(250)).duration_ms == 250


def test_a_wav_cannot_be_built_from_an_encoding_that_is_not_pcm() -> None:
    fmt = AudioFormat(encoding="mulaw", sample_rate=8000, channels=1)

    with pytest.raises(WavError, match="mulaw"):
        build_wav(b"\x00" * 100, fmt)


@pytest.mark.parametrize(
    "data", [b"", b"not audio at all", b"RIFF" + b"\x00" * 8], ids=["empty", "text", "truncated"]
)
def test_reading_a_non_wav_says_so(data: bytes) -> None:
    with pytest.raises(WavError):
        read_wav(data)


# --- validation -----------------------------------------------------------


def test_a_valid_utterance_comes_back_described() -> None:
    audio = validate_utterance(wav_bytes(800))

    assert audio.format == expected_format()
    assert audio.duration_ms == 800
    assert audio.data == wav_bytes(800)


def test_an_empty_payload_is_refused() -> None:
    with pytest.raises(InvalidAudio, match="empty"):
        validate_utterance(b"")


def test_a_payload_that_is_not_a_wav_is_refused() -> None:
    with pytest.raises(InvalidAudio):
        validate_utterance(b"\x00" * 4096)


def test_stereo_is_refused() -> None:
    with pytest.raises(InvalidAudio, match="mono"):
        validate_utterance(wav_bytes(500, channels=2))


def test_eight_kilohertz_is_refused() -> None:
    """Telephony's rate. It arrives with the milestone that needs it."""
    with pytest.raises(InvalidAudio, match="16000 Hz"):
        validate_utterance(wav_bytes(500, sample_rate=8000))


def test_twenty_four_bit_is_refused() -> None:
    with pytest.raises(InvalidAudio, match="16-bit"):
        validate_utterance(wav_bytes(500, sample_width=3))


def test_eight_bit_is_refused() -> None:
    with pytest.raises(InvalidAudio, match="16-bit"):
        validate_utterance(wav_bytes(500, sample_width=1))


def test_an_oversized_payload_is_refused_before_it_is_parsed() -> None:
    """The check is a length comparison, so a hostile frame costs nothing."""
    with pytest.raises(AudioTooLarge, match="at most 1024 bytes"):
        validate_utterance(wav_bytes(5000), max_bytes=1024)


def test_the_size_limit_is_checked_before_the_format() -> None:
    """Otherwise a huge malformed blob would be decoded to be rejected."""
    with pytest.raises(AudioTooLarge):
        validate_utterance(b"\x00" * 5000, max_bytes=1024)


def test_the_accepted_rate_is_configurable() -> None:
    audio = validate_utterance(wav_bytes(500, sample_rate=24000), sample_rate=24000)

    assert audio.format.sample_rate == 24000


def test_the_expected_format_is_the_documented_one() -> None:
    fmt = expected_format()

    assert (fmt.encoding, fmt.sample_rate, fmt.channels, fmt.container) == (
        PCM_S16LE,
        16000,
        1,
        "wav",
    )


def test_a_format_describes_itself_readably() -> None:
    assert str(expected_format()) == "pcm_s16le 16000Hz mono (wav)"
