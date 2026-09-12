"""Voice activity detection: energy, zero crossings, and an adaptive floor."""

import math
import struct

import pytest

from app.audio.vad import (
    MIN_ZERO_CROSSING_RATE,
    VoiceActivityDetector,
    rms,
    zero_crossing_rate,
)

from .conftest import pcm_silence, pcm_tone

FRAME = pcm_tone(9000)
QUIET = pcm_silence()


def _dc(value: int, count: int = 160) -> bytes:
    """A constant offset: loud by energy, and not a sound anybody made."""
    return struct.pack(f"<{count}h", *([value] * count))


def _noise(amplitude: int = 9000, count: int = 160) -> bytes:
    """Alternating samples: the highest zero-crossing rate there is."""
    return struct.pack(
        f"<{count}h", *[amplitude if i % 2 else -amplitude for i in range(count)]
    )


# --- the measurements -----------------------------------------------------


def test_silence_has_no_energy() -> None:
    assert rms(QUIET) == 0.0


def test_a_constant_signal_has_its_own_amplitude() -> None:
    assert rms(_dc(1000)) == pytest.approx(1000.0)


def test_a_sine_has_about_seventy_percent_of_its_peak() -> None:
    """Root-mean-square of a sine is its amplitude over root two."""
    assert rms(pcm_tone(10000)) == pytest.approx(10000 / math.sqrt(2), rel=0.02)


def test_energy_of_nothing_is_zero_rather_than_an_error() -> None:
    assert rms(b"") == 0.0
    assert rms(b"\x00") == 0.0


def test_a_constant_signal_never_crosses_zero() -> None:
    assert zero_crossing_rate(_dc(5000)) == 0.0


def test_alternating_samples_cross_every_time() -> None:
    assert zero_crossing_rate(_noise()) == pytest.approx(1.0)


def test_a_tone_crosses_twice_a_cycle() -> None:
    """220 Hz at 8 kHz is about 0.055 crossings a sample."""
    assert zero_crossing_rate(pcm_tone(9000, hz=220)) == pytest.approx(0.055, abs=0.02)


def test_crossings_of_nothing_are_zero() -> None:
    assert zero_crossing_rate(b"") == 0.0
    assert zero_crossing_rate(b"\x00\x00") == 0.0


# --- the decision ---------------------------------------------------------


def test_loud_speech_is_detected() -> None:
    assert VoiceActivityDetector().is_speech(FRAME)


def test_silence_is_not_speech() -> None:
    assert not VoiceActivityDetector().is_speech(QUIET)


def test_something_quieter_than_the_margin_is_not_speech() -> None:
    assert not VoiceActivityDetector().is_speech(pcm_tone(100))


def test_a_loud_constant_offset_is_not_speech() -> None:
    """A hum can be loud and still cross zero almost never."""
    analysis = VoiceActivityDetector().analyse(_dc(20000))

    assert analysis.energy > 1000
    assert analysis.zero_crossing_rate < MIN_ZERO_CROSSING_RATE
    assert not analysis.speech


def test_the_decision_is_deterministic() -> None:
    first = VoiceActivityDetector()
    second = VoiceActivityDetector()

    assert [first.is_speech(FRAME) for _ in range(5)] == [
        second.is_speech(FRAME) for _ in range(5)
    ]


# --- the adaptive floor ---------------------------------------------------


def test_the_floor_starts_at_nothing() -> None:
    assert VoiceActivityDetector().noise_floor == 0.0


def test_quiet_frames_teach_the_floor() -> None:
    detector = VoiceActivityDetector()
    hiss = pcm_tone(200)

    for _ in range(60):
        detector.analyse(hiss)

    assert detector.noise_floor == pytest.approx(rms(hiss), rel=0.1)


def test_speech_does_not_teach_the_floor() -> None:
    """Learning from the caller would teach the detector to ignore them."""
    detector = VoiceActivityDetector()
    for _ in range(50):
        detector.analyse(FRAME)

    assert detector.noise_floor == 0.0


def test_a_noisy_room_raises_the_bar() -> None:
    """The same margin above a louder room, without anybody retuning it."""
    quiet_room = VoiceActivityDetector()
    noisy_room = VoiceActivityDetector()
    for _ in range(100):
        noisy_room.analyse(pcm_tone(300))  # below the margin, so it is noise

    assert noisy_room.enter_threshold > quiet_room.enter_threshold
    assert noisy_room.noise_floor == pytest.approx(rms(pcm_tone(300)), rel=0.1)


def test_speech_that_beat_a_quiet_room_can_be_lost_in_a_noisy_one() -> None:
    """The cost of adapting: the bar rises for the caller as well as the room."""
    noisy_room = VoiceActivityDetector()
    for _ in range(100):
        noisy_room.analyse(pcm_tone(300))
    barely = pcm_tone(500)

    assert VoiceActivityDetector().is_speech(barely)
    assert not noisy_room.is_speech(barely)


def test_sustained_noise_above_the_margin_is_mistaken_for_speech() -> None:
    """The honest limitation of an energy detector, asserted rather than hidden.

    A television left on in the next room is louder than the margin, so it
    reads as somebody talking and never teaches the floor. A trained detector
    would know the difference; this one cannot, and the README says so.
    """
    detector = VoiceActivityDetector()

    for _ in range(100):
        assert detector.is_speech(pcm_tone(4000))
    assert detector.noise_floor == 0.0


def test_the_floor_can_be_forgotten_between_calls() -> None:
    detector = VoiceActivityDetector()
    for _ in range(100):
        detector.analyse(pcm_tone(2000))

    detector.reset()

    assert detector.noise_floor == 0.0
    assert not detector.in_speech


# --- hysteresis -----------------------------------------------------------


def test_leaving_speech_is_easier_than_entering_it() -> None:
    detector = VoiceActivityDetector()

    assert detector.leave_threshold < detector.enter_threshold


def test_a_brief_quiet_frame_between_words_stays_speech() -> None:
    """Speech is full of gaps; each one is not the end of a sentence."""
    detector = VoiceActivityDetector(hysteresis=0.5)
    detector.analyse(FRAME)

    between = pcm_tone(int(detector.enter_threshold * 0.8))
    assert detector.is_speech(between)


def test_the_same_frame_would_not_have_started_speech() -> None:
    detector = VoiceActivityDetector(hysteresis=0.5)
    between = pcm_tone(int(detector.enter_threshold * 0.8))

    assert not detector.is_speech(between)


@pytest.mark.parametrize(
    ("field", "value"),
    [("noise_floor_alpha", 0.0), ("noise_floor_alpha", 1.5), ("hysteresis", 0.0)],
)
def test_a_setting_that_cannot_work_is_refused(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        VoiceActivityDetector(**{field: value})


def test_the_analysis_reports_what_it_decided_on() -> None:
    analysis = VoiceActivityDetector().analyse(FRAME)

    assert analysis.speech
    assert analysis.energy > analysis.threshold
    assert analysis.threshold == pytest.approx(300.0)
