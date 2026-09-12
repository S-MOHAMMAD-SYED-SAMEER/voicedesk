"""Adding scenarios up, with the denominators showing.

The rule this file mostly protects: a ratio with nothing in its denominator is
`N/A`, never `0%`. Zero percent means "none of the ones that happened", and
saying that about nothing that happened is a claim nobody made.
"""

from decimal import Decimal

from app.evals.checks import (
    HALLUCINATED_AVAILABILITY,
    UNVERIFIED_RESCHEDULE,
    Assessment,
    Finding,
    ToolTally,
)
from app.evals.model import say
from app.evals.scenario import Beat, Expectation, Scenario, ServiceSpec, World, weekdays
from app.evals.scoring import (
    MEANINGFUL_SAMPLE,
    Percentiles,
    Ratio,
    ScenarioResult,
    percentiles,
    score,
)
from app.evals.trace import SCRIPT_EXHAUSTED, CallTrace, TurnTrace

VERSION = "m9-test"


def _scenario(name: str, *, task: str = "book", escalate: bool = False) -> Scenario:
    return Scenario(
        name=name,
        kind="text",
        description="An example.",
        world=World(services=(ServiceSpec("Haircut"),), hours=weekdays()),
        script=(Beat(caller="x", responses=(say("y"),)),),
        expect=Expectation(task=task, should_escalate=escalate),
    )


def _result(
    name: str,
    *,
    findings: tuple[Finding, ...] = (),
    tools: ToolTally | None = None,
    escalated: bool = False,
    should_escalate: bool = False,
    booked_turn: int | None = None,
    cost: Decimal | None = None,
) -> ScenarioResult:
    trace = CallTrace(scenario=name, kind="text")
    trace.total_cost_usd = cost
    if booked_turn is not None:
        for index in range(booked_turn):
            trace.turns.append(
                TurnTrace(
                    index=index,
                    caller_text="x",
                    reply="y",
                    booked_appointment_id="a" if index == booked_turn - 1 else None,
                )
            )
    if escalated:
        if not trace.turns:
            trace.turns.append(TurnTrace(index=0, caller_text="x", reply="y"))
        trace.turns[-1].escalated = True

    assessment = Assessment(scenario=name)
    assessment.findings = list(findings)
    assessment.tools = tools or ToolTally()
    return ScenarioResult(
        scenario=_scenario(name, escalate=should_escalate),
        trace=trace,
        assessment=assessment,
    )


def _finding(category: str) -> Finding:
    return Finding(category=category, expected="a", observed="b")


# --- ratios ----------------------------------------------------------------


def test_a_ratio_reports_its_percentage() -> None:
    assert Ratio(3, 4).percent == 75.0


def test_a_ratio_with_nothing_in_it_is_not_zero_percent() -> None:
    empty = Ratio(0, 0)

    assert empty.defined is False
    assert empty.percent is None
    assert "N/A" in str(empty)


def test_a_ratio_prints_its_denominator() -> None:
    assert "3/4" in str(Ratio(3, 4))


# --- task success ----------------------------------------------------------


def test_task_success_counts_scenarios_with_no_blocking_finding() -> None:
    report = score(
        [_result("a"), _result("b", findings=(_finding("BOOKING_FAILURE"),))],
        VERSION,
    )

    assert report.passed == 1
    assert report.failed == 1
    assert report.task_success.percent == 50.0


def test_a_non_blocking_finding_does_not_fail_a_scenario() -> None:
    report = score([_result("a", findings=(_finding(UNVERIFIED_RESCHEDULE),))], VERSION)

    assert report.passed == 1
    assert report.unverified_reschedules == 1


def test_an_empty_suite_has_no_task_success_rather_than_zero() -> None:
    assert score([], VERSION).task_success.defined is False


def test_an_authoring_failure_is_counted_apart() -> None:
    """Still a failure, but the dataset's rather than the receptionist's."""
    report = score([_result("a", findings=(_finding(SCRIPT_EXHAUSTED),))], VERSION)

    assert report.failed == 1
    assert report.authoring_failures == 1
    assert SCRIPT_EXHAUSTED not in report.behavioural_failures


# --- hallucination and reschedules -----------------------------------------


def test_hallucinations_are_counted_not_rated() -> None:
    report = score(
        [
            _result("a", findings=(_finding(HALLUCINATED_AVAILABILITY),)),
            _result("b", findings=(_finding(HALLUCINATED_AVAILABILITY),)),
        ],
        VERSION,
    )

    assert report.hallucinated_availability == 2


def test_a_clean_suite_reports_zero_hallucinations() -> None:
    assert score([_result("a")], VERSION).hallucinated_availability == 0


# --- tool correctness ------------------------------------------------------


def test_tool_correctness_sums_across_scenarios() -> None:
    report = score(
        [
            _result("a", tools=ToolTally(correct=2)),
            _result("b", tools=ToolTally(correct=1, wrong=1)),
        ],
        VERSION,
    )

    assert report.tools.correct == 3
    assert report.tools.wrong == 1
    assert report.tool_correctness.percent == 75.0


def test_tool_correctness_with_no_tools_is_not_zero_percent() -> None:
    assert score([_result("a")], VERSION).tool_correctness.defined is False


def test_every_tool_outcome_is_in_the_denominator() -> None:
    tally = ToolTally(correct=1, wrong=1, missing=1, duplicate=1, invalid_argument=1)

    assert tally.total == 5


# --- escalation ------------------------------------------------------------


def test_escalation_precision_is_of_those_that_escalated() -> None:
    report = score(
        [
            _result("a", escalated=True, should_escalate=True),
            _result("b", escalated=True, should_escalate=False),
            _result("c", escalated=False, should_escalate=False),
        ],
        VERSION,
    )

    assert report.escalation_precision == Ratio(1, 2)


def test_escalation_recall_is_of_those_that_should_have() -> None:
    report = score(
        [
            _result("a", escalated=True, should_escalate=True),
            _result("b", escalated=False, should_escalate=True),
        ],
        VERSION,
    )

    assert report.escalation_recall == Ratio(1, 2)


def test_precision_with_no_escalations_is_not_zero() -> None:
    report = score([_result("a")], VERSION)

    assert report.escalation_precision.defined is False


def test_recall_with_nothing_required_is_not_zero() -> None:
    report = score([_result("a", escalated=True)], VERSION)

    assert report.escalation_recall.defined is False


def test_perfect_escalation_is_reported_as_such() -> None:
    report = score(
        [_result("a", escalated=True, should_escalate=True)], VERSION
    )

    assert report.escalation_precision == Ratio(1, 1)
    assert report.escalation_recall == Ratio(1, 1)


# --- turns to booking ------------------------------------------------------


def test_turns_to_booking_counts_only_calls_that_booked() -> None:
    report = score(
        [_result("a", booked_turn=2), _result("b"), _result("c", booked_turn=4)],
        VERSION,
    )

    assert report.turns_to_booking.count == 2


def test_turns_to_booking_is_the_turn_the_booking_happened_on() -> None:
    report = score([_result("a", booked_turn=3)], VERSION)

    assert report.turns_to_booking.p50 == 3


def test_no_booking_means_nothing_to_measure() -> None:
    assert score([_result("a")], VERSION).turns_to_booking.count == 0


# --- percentiles -----------------------------------------------------------


def test_no_values_give_no_percentiles() -> None:
    assert percentiles([]) == Percentiles(0, None, None)


def test_one_value_is_its_own_median_and_its_own_p95() -> None:
    assert percentiles([4.0]) == Percentiles(1, 4.0, 4.0)


def test_percentiles_of_a_small_sample_are_marked_unmeaningful() -> None:
    assert percentiles([1.0, 2.0, 3.0]).meaningful is False


def test_a_large_enough_sample_is_meaningful() -> None:
    assert percentiles([1.0] * MEANINGFUL_SAMPLE).meaningful is True


def test_the_median_is_the_median() -> None:
    assert percentiles([1.0, 2.0, 3.0, 4.0, 5.0]).p50 == 3.0


# --- cost ------------------------------------------------------------------


def test_a_call_with_no_total_is_unpriced_not_free() -> None:
    report = score([_result("a", cost=None)], VERSION)

    assert report.measured_calls == 0
    assert report.unpriced_calls == 1
    assert report.total_cost_usd is None


def test_priced_calls_are_summed() -> None:
    report = score(
        [
            _result("a", cost=Decimal("0.010000")),
            _result("b", cost=Decimal("0.020000")),
        ],
        VERSION,
    )

    assert report.measured_calls == 2
    assert report.total_cost_usd == Decimal("0.030000")


def test_a_mixed_suite_reports_both_counts() -> None:
    report = score([_result("a", cost=Decimal("0.01")), _result("b")], VERSION)

    assert (report.measured_calls, report.unpriced_calls) == (1, 1)


# --- findings --------------------------------------------------------------


def test_findings_are_counted_by_category() -> None:
    report = score(
        [
            _result("a", findings=(_finding("WRONG_TOOL"), _finding("WRONG_TOOL"))),
            _result("b", findings=(_finding("MISSING_TOOL"),)),
        ],
        VERSION,
    )

    assert report.findings_by_category == {"MISSING_TOOL": 1, "WRONG_TOOL": 2}


def test_the_observation_is_kept_out_of_the_behavioural_breakdown() -> None:
    report = score([_result("a", findings=(_finding(UNVERIFIED_RESCHEDULE),))], VERSION)

    assert UNVERIFIED_RESCHEDULE in report.findings_by_category
    assert UNVERIFIED_RESCHEDULE not in report.behavioural_failures


def test_the_report_carries_its_version() -> None:
    assert score([], VERSION).version == VERSION
