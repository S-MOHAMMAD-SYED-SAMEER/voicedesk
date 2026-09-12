"""Deciding when the caller has stopped — which is not the same as silence."""

import pytest

from app.audio.endpoint import Endpointer, SpeechEnded, SpeechStarted
from app.audio.vad import VoiceActivityDetector

from .conftest import pcm_silence, pcm_tone

LOUD = pcm_tone(9000)
QUIET = pcm_silence()
FRAME_MS = 20


def _endpointer(**overrides) -> Endpointer:
    fields = {
        "sample_rate": 8000,
        "min_speech_ms": 120,
        "silence_ms": 700,
        "max_utterance_ms": 20_000,
        "preroll_ms": 200,
    }
    fields.update(overrides)
    return Endpointer(VoiceActivityDetector(), **fields)


def _feed(endpointer: Endpointer, frame: bytes, count: int) -> list:
    events = []
    for _ in range(count):
        events.extend(endpointer.feed(frame))
    return events


def _say(endpointer: Endpointer, speech_frames: int = 10, silence_frames: int = 40):
    return _feed(endpointer, LOUD, speech_frames) + _feed(
        endpointer, QUIET, silence_frames
    )


# --- a normal utterance ---------------------------------------------------


def test_speech_then_a_pause_is_one_utterance() -> None:
    events = _say(_endpointer())

    assert [type(event).__name__ for event in events] == [
        "SpeechStarted",
        "SpeechEnded",
    ]


def test_the_utterance_carries_its_audio() -> None:
    ended = [event for event in _say(_endpointer()) if isinstance(event, SpeechEnded)][0]

    assert ended.audio
    assert not ended.truncated
    assert ended.duration_ms > 0


def test_speech_starts_before_the_caller_has_finished() -> None:
    """Barge-in depends on knowing early, not at the end of the sentence."""
    events = _feed(_endpointer(), LOUD, 10)

    assert [type(event).__name__ for event in events] == ["SpeechStarted"]


def test_the_detector_reports_that_it_is_listening() -> None:
    endpointer = _endpointer()
    _feed(endpointer, LOUD, 10)

    assert endpointer.speaking


# --- the minimum ----------------------------------------------------------


def test_a_click_is_not_an_utterance() -> None:
    """Two frames is 40 ms. Nothing reaches a model because of a line pop."""
    events = _feed(_endpointer(), LOUD, 2) + _feed(_endpointer(), QUIET, 50)

    assert events == []


def test_two_separate_clicks_do_not_add_up() -> None:
    """Speech has to be continuous to count."""
    endpointer = _endpointer()
    events = _feed(endpointer, LOUD, 3)
    events += _feed(endpointer, QUIET, 10)
    events += _feed(endpointer, LOUD, 3)

    assert events == []


def test_the_minimum_is_configurable() -> None:
    events = _feed(_endpointer(min_speech_ms=40), LOUD, 2)

    assert [type(event).__name__ for event in events] == ["SpeechStarted"]


# --- pauses ---------------------------------------------------------------


def test_a_short_pause_does_not_end_the_turn() -> None:
    """Breathing mid-sentence is not the end of a sentence."""
    endpointer = _endpointer()
    events = _feed(endpointer, LOUD, 10)
    events += _feed(endpointer, QUIET, 20)  # 400 ms, under the 700 ms boundary
    events += _feed(endpointer, LOUD, 10)

    assert [type(event).__name__ for event in events] == ["SpeechStarted"]
    assert endpointer.speaking


def test_a_long_pause_ends_the_turn() -> None:
    endpointer = _endpointer()
    _feed(endpointer, LOUD, 10)
    events = _feed(endpointer, QUIET, 35)  # exactly 700 ms

    assert len(events) == 1
    assert isinstance(events[0], SpeechEnded)
    assert not endpointer.speaking


def test_the_pause_is_configurable() -> None:
    endpointer = _endpointer(silence_ms=100)
    _feed(endpointer, LOUD, 10)
    events = _feed(endpointer, QUIET, 5)

    assert len(events) == 1


# --- the ceiling ----------------------------------------------------------


def test_a_caller_who_never_pauses_is_answered_anyway() -> None:
    endpointer = _endpointer(max_utterance_ms=400)

    events = _feed(endpointer, LOUD, 40)

    ended = [event for event in events if isinstance(event, SpeechEnded)]
    assert ended
    assert ended[0].truncated


def test_the_ceiling_bounds_the_buffer() -> None:
    """An unbounded buffer on an open socket is a bug, not a feature."""
    endpointer = _endpointer(max_utterance_ms=400)

    _feed(endpointer, LOUD, 200)

    assert endpointer.utterance_ms <= 400


# --- the pre-roll ---------------------------------------------------------


def test_audio_from_before_the_decision_is_kept() -> None:
    """The previous milestone threw this away and clipped every first word.

    The window is the last `preroll_ms` before the detector was convinced, so
    it necessarily contains the run-up — including the 120 ms of speech that
    did the convincing. Here that is 200 ms of pre-roll, the 80 ms of speech
    after the decision, and the 700 ms of silence that ended it.
    """
    endpointer = _endpointer()
    _feed(endpointer, QUIET, 20)

    events = _feed(endpointer, LOUD, 10) + _feed(endpointer, QUIET, 35)

    ended = [event for event in events if isinstance(event, SpeechEnded)][0]
    assert ended.duration_ms == pytest.approx(980, abs=FRAME_MS)


def test_the_preroll_does_not_grow_without_limit() -> None:
    """A caller silent for a minute costs a hundred milliseconds of buffer."""
    endpointer = _endpointer(preroll_ms=100)
    _feed(endpointer, QUIET, 200)

    events = _feed(endpointer, LOUD, 10) + _feed(endpointer, QUIET, 35)

    ended = [event for event in events if isinstance(event, SpeechEnded)][0]
    assert ended.duration_ms == pytest.approx(880, abs=FRAME_MS)


def test_without_a_preroll_the_first_syllable_is_lost() -> None:
    """Which is the whole reason the pre-roll exists.

    With none, the 120 ms of speech that proved somebody was talking is gone
    by the time anybody is listening, and only what came after it survives.
    """
    endpointer = _endpointer(preroll_ms=0)
    _feed(endpointer, QUIET, 20)

    events = _feed(endpointer, LOUD, 10) + _feed(endpointer, QUIET, 35)

    ended = [event for event in events if isinstance(event, SpeechEnded)][0]
    assert ended.duration_ms == pytest.approx(780, abs=FRAME_MS)
    # 200 ms was spoken; 80 ms of it survived.
    assert ended.duration_ms < 980


# --- nothing said ---------------------------------------------------------


def test_silence_alone_produces_nothing_at_all() -> None:
    """No event means no recognition, which means no model call."""
    assert _feed(_endpointer(), QUIET, 500) == []


def test_a_quiet_room_never_fills_memory() -> None:
    endpointer = _endpointer()
    _feed(endpointer, QUIET, 5000)

    assert endpointer.utterance_ms == 0


def test_an_empty_frame_is_ignored() -> None:
    assert _endpointer().feed(b"") == []


# --- ending mid-sentence --------------------------------------------------


def test_flushing_returns_whatever_was_being_said() -> None:
    """The call is ending; the half-sentence still deserves an answer."""
    endpointer = _endpointer()
    _feed(endpointer, LOUD, 10)

    flushed = endpointer.flush()

    assert isinstance(flushed, SpeechEnded)
    assert flushed.truncated
    assert not endpointer.speaking


def test_flushing_nothing_returns_nothing() -> None:
    assert _endpointer().flush() is None


def test_resetting_forgets_the_utterance() -> None:
    endpointer = _endpointer()
    _feed(endpointer, LOUD, 10)

    endpointer.reset()

    assert not endpointer.speaking
    assert endpointer.utterance_ms == 0


def test_a_second_utterance_follows_the_first() -> None:
    endpointer = _endpointer()
    events = _say(endpointer) + _say(endpointer)

    assert [type(event).__name__ for event in events] == [
        "SpeechStarted",
        "SpeechEnded",
        "SpeechStarted",
        "SpeechEnded",
    ]
