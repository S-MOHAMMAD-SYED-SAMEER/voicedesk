"""Latency reporting: what is measured, and when not to believe it."""

import pytest
from sqlalchemy.orm import Session

from app.metrics import (
    MEANINGFUL_SAMPLE,
    Summary,
    collect,
    first_audio_latency_ms,
    percentiles,
    render,
    report,
    summarise,
    turns_with_role,
)
from app.models import Call, Turn, TurnRole


def _turn(call: Call, **fields) -> Turn:
    defaults = {
        "call_id": call.id,
        "role": TurnRole.AGENT,
        "text": "We're open until five.",
    }
    defaults.update(fields)
    return Turn(**defaults)


# --- the arithmetic -------------------------------------------------------


def test_nothing_has_no_percentiles() -> None:
    assert percentiles([]) == (None, None)


def test_one_sample_is_its_own_median_and_its_own_p95() -> None:
    """Worth reporting; not worth interpolating."""
    assert percentiles([420]) == (420, 420)


def test_the_median_is_the_middle() -> None:
    p50, _ = percentiles([10, 20, 30, 40, 50])

    assert p50 == 30


def test_the_ninety_fifth_is_near_the_top() -> None:
    _, p95 = percentiles(list(range(1, 101)))

    assert 94 <= p95 <= 96


def test_order_does_not_matter() -> None:
    assert percentiles([50, 10, 30]) == percentiles([10, 30, 50])


def test_a_small_sample_is_flagged_rather_than_dressed_up() -> None:
    assert not summarise("x", [1, 2, 3]).meaningful
    assert summarise("x", list(range(MEANINGFUL_SAMPLE))).meaningful


# --- what first-audio latency means ---------------------------------------


def test_first_audio_latency_is_the_three_stages_added_up(
    session: Session, call: Call
) -> None:
    """Caller stops → recognition → model and tools → first audio."""
    turn = _turn(call, stt_latency_ms=120, llm_latency_ms=600, tts_latency_ms=180)

    assert first_audio_latency_ms(turn) == 900


@pytest.mark.parametrize(
    "missing", ["stt_latency_ms", "llm_latency_ms", "tts_latency_ms"]
)
def test_a_turn_missing_a_stage_is_not_measured(
    session: Session, call: Call, missing: str
) -> None:
    """A partial sum would read as a fast turn, which is worse than no turn."""
    fields = {"stt_latency_ms": 120, "llm_latency_ms": 600, "tts_latency_ms": 180}
    fields[missing] = None

    assert first_audio_latency_ms(_turn(call, **fields)) is None


def test_rows_from_before_realtime_are_skipped_not_counted_as_zero(
    session: Session, call: Call
) -> None:
    """Every turn the earlier milestones wrote has null latencies."""
    session.add(_turn(call, llm_latency_ms=600))
    session.commit()

    assert collect(session)["first_audio_latency_ms"] == []


# --- gathering ------------------------------------------------------------


def test_measured_turns_are_collected(session: Session, call: Call) -> None:
    session.add_all(
        [
            _turn(call, stt_latency_ms=100, llm_latency_ms=500, tts_latency_ms=100),
            _turn(call, stt_latency_ms=200, llm_latency_ms=500, tts_latency_ms=200),
        ]
    )
    session.commit()

    assert collect(session)["first_audio_latency_ms"] == [700, 900]


def test_how_long_the_caller_spoke_is_collected(
    session: Session, call: Call
) -> None:
    """From the caller's own row, which is where that number is written."""
    session.add(_turn(call, role=TurnRole.CALLER, text="hello", audio_ms=1500))
    session.add(_turn(call, stt_latency_ms=1, llm_latency_ms=2, tts_latency_ms=3))
    session.commit()

    measured = collect(session)

    assert measured["caller_audio_ms"] == [1500]
    assert measured["first_audio_latency_ms"] == [6]


def test_the_two_metrics_come_from_the_two_roles(
    session: Session, call: Call
) -> None:
    session.add(_turn(call, role=TurnRole.CALLER, text="hello", audio_ms=1500))
    session.commit()

    assert len(turns_with_role(session, TurnRole.CALLER)) == 1
    assert turns_with_role(session, TurnRole.AGENT) == []


def test_one_call_can_be_asked_about(session: Session, call: Call) -> None:
    other = Call(direction=call.direction, from_number="+441", to_number="+442")
    session.add(other)
    session.commit()
    session.add(_turn(call, stt_latency_ms=100, llm_latency_ms=100, tts_latency_ms=100))
    session.add(
        _turn(other, stt_latency_ms=900, llm_latency_ms=900, tts_latency_ms=900)
    )
    session.commit()

    assert collect(session, call.id)["first_audio_latency_ms"] == [300]


# --- the report -----------------------------------------------------------


def test_the_table_has_a_row_for_each_metric() -> None:
    body = render(
        [summarise("first_audio_latency_ms", [100]), summarise("caller_audio_ms", [])]
    )

    assert "first_audio_latency_ms" in body
    assert "caller_audio_ms" in body
    assert "p50" in body and "p95" in body


def test_a_metric_with_no_samples_says_so_rather_than_zero() -> None:
    """Zero milliseconds and "never measured" must not look the same."""
    row = render([summarise("first_audio_latency_ms", [])]).split("\n")[2]

    assert row.count("n/a") == 2
    assert row.split() == ["first_audio_latency_ms", "0", "n/a", "n/a"]


def test_a_thin_sample_is_marked_in_the_table() -> None:
    body = render([summarise("first_audio_latency_ms", [100, 200])])

    assert "*" in body
    assert "not evidence" in body


def test_a_full_sample_is_not_marked() -> None:
    body = render([summarise("x", [100] * MEANINGFUL_SAMPLE)])

    assert "not evidence" not in body


def test_an_empty_database_explains_itself(session: Session) -> None:
    body = report(session)

    assert "No turn carries latency yet" in body


def test_the_report_runs_against_real_rows(session: Session, call: Call) -> None:
    session.add_all(
        [
            _turn(call, stt_latency_ms=100, llm_latency_ms=400, tts_latency_ms=100)
            for _ in range(3)
        ]
    )
    session.commit()

    body = report(session)

    assert "600" in body
    assert "No turn carries latency yet" not in body


def test_a_summary_knows_whether_to_be_believed() -> None:
    assert not Summary("x", 3, 1.0, 1.0).meaningful
    assert Summary("x", 100, 1.0, 1.0).meaningful
