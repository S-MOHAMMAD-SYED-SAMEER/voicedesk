"""G.711 µ-law and the 2:1 resampler, written rather than imported.

Python 3.13 removed `audioop`, so these are the project's own. The domain is
256 values wide and the rate change is exactly 2:1, which means every claim
here can be checked rather than trusted.
"""

import struct

import pytest

from app.audio.telephony import (
    CLIP,
    FRAME_BYTES,
    TelephonyAudioError,
    downsample_16k_to_8k,
    frames,
    mean_amplitude,
    mulaw_decode,
    mulaw_encode,
    speech_to_mulaw,
    telephony_format,
    upsample_8k_to_16k,
    utterance_from_mulaw,
)
from app.audio.format import validate_utterance
from app.providers.speech import PCM_S16LE, AudioFormat, build_wav, read_wav
from app.providers.tts import Speech

SILENCE_CODE = 0xFF
NEGATIVE_ZERO_CODE = 0x7F


def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _samples(pcm: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(pcm) // 2}h", pcm)


def _speech(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> Speech:
    fmt = AudioFormat(PCM_S16LE, sample_rate, channels, "wav")
    return Speech(audio=build_wav(pcm, fmt), format=fmt)


# --- µ-law anchors --------------------------------------------------------


def test_silence_encodes_to_the_standard_silence_code() -> None:
    assert mulaw_encode(_pcm(0)) == bytes([SILENCE_CODE])


def test_the_silence_code_decodes_to_silence() -> None:
    assert mulaw_decode(bytes([SILENCE_CODE])) == _pcm(0)


def test_a_positive_sample_keeps_its_sign() -> None:
    """µ-law stores the byte inverted, so a positive sample sets the top bit."""
    code = mulaw_encode(_pcm(8000))[0]

    assert code & 0x80
    assert _samples(mulaw_decode(bytes([code])))[0] > 0


def test_a_negative_sample_keeps_its_sign() -> None:
    code = mulaw_encode(_pcm(-8000))[0]

    assert not code & 0x80
    assert _samples(mulaw_decode(bytes([code])))[0] < 0


def test_a_sample_and_its_negation_differ_only_in_sign() -> None:
    positive = _samples(mulaw_decode(mulaw_encode(_pcm(8000))))[0]
    negative = _samples(mulaw_decode(mulaw_encode(_pcm(-8000))))[0]

    assert positive == -negative


def test_the_loudest_positive_sample_clips_to_the_extreme_code() -> None:
    assert mulaw_encode(_pcm(32767)) == bytes([0x80])
    assert mulaw_encode(_pcm(CLIP)) == bytes([0x80])


def test_the_loudest_negative_sample_clips_to_the_extreme_code() -> None:
    assert mulaw_encode(_pcm(-32768)) == bytes([0x00])


def test_every_code_decodes_and_re_encodes_to_itself_except_negative_zero() -> None:
    """µ-law has two codes for zero; one of them is not a round trip.

    `0x7F` is negative zero. It decodes to 0, and 0 encodes to the positive
    zero code `0xFF`. That is the standard's behaviour, not a defect, and a
    test demanding all 256 values survive would be wrong rather than strict.
    """
    survivors = {
        code
        for code in range(256)
        if mulaw_encode(mulaw_decode(bytes([code]))) == bytes([code])
    }

    assert set(range(256)) - survivors == {NEGATIVE_ZERO_CODE}


def test_negative_zero_becomes_positive_zero() -> None:
    assert mulaw_decode(bytes([NEGATIVE_ZERO_CODE])) == _pcm(0)
    assert mulaw_encode(mulaw_decode(bytes([NEGATIVE_ZERO_CODE]))) == bytes(
        [SILENCE_CODE]
    )


def test_encoding_is_monotonic_in_loudness() -> None:
    quiet = abs(_samples(mulaw_decode(mulaw_encode(_pcm(1000))))[0])
    loud = abs(_samples(mulaw_decode(mulaw_encode(_pcm(20000))))[0])

    assert loud > quiet


def test_the_codec_is_lossy_and_bounded() -> None:
    """µ-law is logarithmic: coarse when loud, fine when quiet. By design."""
    worst = max(
        abs(_samples(mulaw_decode(mulaw_encode(_pcm(value))))[0] - min(value, CLIP))
        for value in range(0, 32768, 97)
    )

    assert worst <= 512


# --- µ-law edges ----------------------------------------------------------


def test_empty_input_encodes_and_decodes_to_nothing() -> None:
    assert mulaw_encode(b"") == b""
    assert mulaw_decode(b"") == b""


def test_a_partial_sample_is_refused_rather_than_dropped() -> None:
    """Half a sample means something upstream is wrong."""
    with pytest.raises(TelephonyAudioError, match="whole number"):
        mulaw_encode(b"\x00")


def test_encoding_produces_one_byte_per_sample() -> None:
    assert len(mulaw_encode(_pcm(*range(0, 300, 3)))) == 100


def test_decoding_produces_two_bytes_per_sample() -> None:
    assert len(mulaw_decode(bytes(100))) == 200


def test_encoding_is_deterministic() -> None:
    pcm = _pcm(0, 500, -500, 12000, -12000)

    assert mulaw_encode(pcm) == mulaw_encode(pcm)


# --- resampling -----------------------------------------------------------


def test_upsampling_doubles_the_samples() -> None:
    assert len(_samples(upsample_8k_to_16k(_pcm(1, 2, 3, 4)))) == 8


def test_downsampling_halves_the_samples() -> None:
    assert len(_samples(downsample_16k_to_8k(_pcm(1, 2, 3, 4, 5, 6)))) == 3


def test_upsampling_interpolates_between_neighbours() -> None:
    assert _samples(upsample_8k_to_16k(_pcm(0, 100))) == (0, 50, 100, 100)


def test_downsampling_averages_pairs() -> None:
    assert _samples(downsample_16k_to_8k(_pcm(0, 100, 200, 400))) == (50, 300)


def test_silence_survives_both_directions() -> None:
    silence = _pcm(0, 0, 0, 0)

    assert upsample_8k_to_16k(silence) == _pcm(0, 0, 0, 0, 0, 0, 0, 0)
    assert downsample_16k_to_8k(silence) == _pcm(0, 0)


def test_a_constant_signal_survives_both_directions() -> None:
    constant = _pcm(*([1234] * 8))

    assert set(_samples(upsample_8k_to_16k(constant))) == {1234}
    assert set(_samples(downsample_16k_to_8k(constant))) == {1234}


def test_a_round_trip_returns_the_original_length() -> None:
    original = _pcm(*range(0, 800, 8))

    assert len(downsample_16k_to_8k(upsample_8k_to_16k(original))) == len(original)


def test_an_odd_number_of_samples_loses_the_leftover_rather_than_raising() -> None:
    """Half a pair is not a sample, and at 16 kHz it is 62 microseconds."""
    assert len(_samples(downsample_16k_to_8k(_pcm(1, 2, 3, 4, 5)))) == 2


def test_resampling_empty_input_gives_nothing() -> None:
    assert upsample_8k_to_16k(b"") == b""
    assert downsample_16k_to_8k(b"") == b""


@pytest.mark.parametrize("convert", [upsample_8k_to_16k, downsample_16k_to_8k])
def test_a_partial_sample_is_refused_by_both_resamplers(convert) -> None:
    with pytest.raises(TelephonyAudioError, match="whole number"):
        convert(b"\x00")


def test_resampling_is_deterministic() -> None:
    pcm = _pcm(*range(-100, 100, 7))

    assert upsample_8k_to_16k(pcm) == upsample_8k_to_16k(pcm)
    assert downsample_16k_to_8k(pcm) == downsample_16k_to_8k(pcm)


# --- the composed conversions ---------------------------------------------


def test_an_utterance_becomes_exactly_what_the_audio_layer_accepts() -> None:
    """The same `Audio` the browser produces. That is the point of all this."""
    audio = utterance_from_mulaw(bytes([SILENCE_CODE]) * 8000)

    assert audio.format.sample_rate == 16000
    assert audio.format.channels == 1
    assert audio.format.encoding == PCM_S16LE
    assert audio.format.container == "wav"
    # And the milestone-5 door accepts it without being told anything.
    assert validate_utterance(audio.data).duration_ms == 1000


def test_an_utterance_reports_the_duration_the_carrier_sent() -> None:
    """8000 bytes of µ-law is one second at 8 kHz, whatever it resamples to."""
    assert utterance_from_mulaw(bytes(4000)).duration_ms == 500


def test_an_empty_utterance_is_refused() -> None:
    with pytest.raises(TelephonyAudioError, match="no audio"):
        utterance_from_mulaw(b"")


def test_a_reply_becomes_telephone_audio() -> None:
    mulaw = speech_to_mulaw(_speech(_pcm(*([0] * 3200))))

    assert len(mulaw) == 1600  # 3200 samples at 16 kHz → 1600 at 8 kHz
    assert set(mulaw) == {SILENCE_CODE}


def test_a_reply_already_at_the_carrier_rate_passes_through() -> None:
    mulaw = speech_to_mulaw(_speech(_pcm(*([0] * 800)), sample_rate=8000))

    assert len(mulaw) == 800


def test_a_reply_at_an_unhandled_rate_is_refused() -> None:
    """Rather than resampled by a ratio this module does not implement."""
    with pytest.raises(TelephonyAudioError, match="44100"):
        speech_to_mulaw(_speech(_pcm(0, 0), sample_rate=44100))


def test_a_stereo_reply_is_refused() -> None:
    with pytest.raises(TelephonyAudioError, match="mono"):
        speech_to_mulaw(_speech(_pcm(0, 0, 0, 0), channels=2))


def test_a_loud_reply_survives_the_whole_round_trip() -> None:
    """Lossy, but recognisably the same sound."""
    original = _pcm(*[int(8000 * (1 if index % 4 < 2 else -1)) for index in range(320)])

    back = _samples(mulaw_decode(speech_to_mulaw(_speech(original))))

    assert len(back) == 160
    assert max(abs(sample) for sample in back) > 7000


# --- framing --------------------------------------------------------------


def test_audio_is_cut_into_carrier_sized_frames() -> None:
    assert [len(frame) for frame in frames(bytes(400))] == [160, 160, 80]


def test_a_frame_is_twenty_milliseconds() -> None:
    assert FRAME_BYTES == 160


def test_a_short_final_frame_is_kept_rather_than_padded() -> None:
    """Padding would be silence the caller waits through."""
    assert b"".join(frames(bytes(401))) == bytes(401)


def test_framing_nothing_gives_no_frames() -> None:
    assert frames(b"") == []


def test_the_carrier_format_describes_itself() -> None:
    fmt = telephony_format()

    assert (fmt.encoding, fmt.sample_rate, fmt.channels) == ("mulaw", 8000, 1)


# --- the amplitude measure ------------------------------------------------


def test_silence_measures_zero() -> None:
    assert mean_amplitude(_pcm(0, 0, 0)) == 0.0


def test_loudness_is_the_mean_absolute_sample() -> None:
    assert mean_amplitude(_pcm(-100, 100, -300, 300)) == 200.0


def test_measuring_nothing_is_zero_rather_than_an_error() -> None:
    assert mean_amplitude(b"") == 0.0
    assert mean_amplitude(b"\x00") == 0.0
