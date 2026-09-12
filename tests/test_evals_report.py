"""What the report says, and what it refuses to say.

The headline number is the one most likely to be quoted out of context, so
these tests are mostly about the sentences around it: the denominators, the
"arithmetic, not evidence" marker, and the line saying what a scripted-model
suite does and does not measure.
"""

import json
from decimal import Decimal

from app.evals.checks import (
    HALLUCINATED_AVAILABILITY,
    UNVERIFIED_RESCHEDULE,
    Assessment,
    Finding,
    ToolTally,
)
from app.evals.model import say
from app.evals.report import HONESTY, as_json, render, to_dict
from app.evals.scenario import (
    AppointmentSpec,
    Beat,
    Expectation,
    Scenario,
    ServiceSpec,
    World,
    weekdays,
)
from app.evals.scoring import ScenarioResult, score
from app.evals.trace import SCRIPT_EXHAUSTED, CallTrace, TurnTrace

VERSION = "m9-test"
TEN = "2026-03-02T10:00:00+00:00"


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
    name: str = "example",
    *,
    findings: tuple[Finding, ...] = (),
    task: str = "book",
    escalate: bool = False,
    booked: bool = False,
    cost: Decimal | None = None,
    appointments: tuple = (),
) -> ScenarioResult:
    trace = CallTrace(scenario=name, kind="text")
    trace.total_cost_usd = cost
    trace.final_appointments = list(appointments)
    trace.turns = [
        TurnTrace(
            index=0,
            caller_text="x",
            reply="y",
            booked_appointment_id="a" if booked else None,
        )
    ]
    assessment = Assessment(scenario=name)
    assessment.findings = list(findings)
    assessment.tools = ToolTally(correct=2)
    return ScenarioResult(
        scenario=_scenario(name, task=task, escalate=escalate),
        trace=trace,
        assessment=assessment,
    )


def _finding(category: str, **overrides) -> Finding:
    fields = {
        "category": category,
        "expected": "a booking",
        "observed": "no booking",
        "evidence": "the appointments table",
    }
    fields.update(overrides)
    return Finding(**fields)


# --- the headline ----------------------------------------------------------


def test_the_report_names_its_version() -> None:
    assert f"VoiceDesk Evaluation — {VERSION}" in render(score([], VERSION))


def test_the_headline_counts_are_printed() -> None:
    text = render(score([_result("a"), _result("b")], VERSION))

    assert "Scenarios: 2" in text
    assert "Passed: 2" in text
    assert "Failed: 0" in text


def test_task_success_shows_its_denominator() -> None:
    text = render(score([_result("a"), _result("b")], VERSION))

    assert "Task success: 100.0% (2/2)" in text


def test_hallucinations_are_a_count_not_a_rate() -> None:
    text = render(
        score([_result("a", findings=(_finding(HALLUCINATED_AVAILABILITY),))], VERSION)
    )

    assert "Hallucinated availability: 1" in text


def test_every_tool_outcome_is_broken_out() -> None:
    text = render(score([_result("a")], VERSION))

    for line in ("correct:", "wrong:", "missing:", "duplicate:", "invalid_argument:"):
        assert line in text


def test_escalation_is_reported_as_a_fraction() -> None:
    text = render(score([_result("a", escalate=True)], VERSION))

    assert "precision:" in text
    assert "recall:" in text


def test_an_undefined_ratio_says_so_rather_than_zero() -> None:
    text = render(score([_result("a")], VERSION))

    assert "N/A" in text
    assert "precision: 0.0%" not in text


# --- honesty ---------------------------------------------------------------


def test_the_report_says_what_it_does_not_measure() -> None:
    text = render(score([_result("a")], VERSION))

    assert HONESTY in text
    assert "no real model's conversational quality" in text


def test_the_honesty_line_mentions_the_scripted_model() -> None:
    assert "Scripted model" in HONESTY


def test_percentiles_of_a_small_sample_are_marked() -> None:
    text = render(score([_result("a", booked=True)], VERSION))

    assert "arithmetic, not evidence" in text


def test_the_reschedule_observation_is_explained_when_it_happens() -> None:
    text = render(
        score([_result("a", findings=(_finding(UNVERIFIED_RESCHEDULE),))], VERSION)
    )

    assert "Unverified reschedules: 1" in text
    assert "not counted against task success" in text


def test_the_explanation_is_absent_when_there_is_nothing_to_explain() -> None:
    text = render(score([_result("a")], VERSION))

    assert "Unverified reschedules: 0" in text
    assert "not counted against task success" not in text


def test_unpriced_calls_are_explained_as_unknown_rather_than_free() -> None:
    text = render(score([_result("a", cost=None)], VERSION))

    assert "unpriced calls: 1" in text
    assert "not that it was free" in text


def test_a_priced_suite_reports_its_total() -> None:
    text = render(score([_result("a", cost=Decimal("0.001234"))], VERSION))

    assert "measured calls: 1" in text
    assert "0.001234 USD" in text


def test_an_authoring_failure_is_labelled_as_the_datasets_fault() -> None:
    text = render(score([_result("a", findings=(_finding(SCRIPT_EXHAUSTED),))], VERSION))

    assert "Scenario-authoring failures: 1" in text
    assert "not in VoiceDesk" in text


# --- per scenario ----------------------------------------------------------


def test_a_passing_scenario_is_listed_as_pass() -> None:
    text = render(score([_result("booking_available")], VERSION))

    assert "PASS booking_available" in text


def test_a_failing_scenario_is_listed_with_its_finding() -> None:
    text = render(
        score(
            [_result("unavailable_slot", findings=(_finding(HALLUCINATED_AVAILABILITY),))],
            VERSION,
        )
    )

    assert "FAIL unavailable_slot" in text
    assert f"finding: {HALLUCINATED_AVAILABILITY}" in text
    assert "evidence: the appointments table" in text


def test_a_scenario_shows_what_was_expected_and_what_happened() -> None:
    text = render(score([_result("a", booked=True)], VERSION))

    assert "expected: book" in text
    assert "observed: book" in text


def test_a_scenario_lists_its_appointments() -> None:
    text = render(
        score(
            [_result("a", appointments=(AppointmentSpec("Haircut", "Ada", TEN),))],
            VERSION,
        )
    )

    assert "Haircut for Ada" in text


def test_a_scenario_with_no_appointments_says_none() -> None:
    assert "appointments: none" in render(score([_result("a")], VERSION))


def test_a_non_blocking_finding_is_shown_as_an_observation() -> None:
    text = render(
        score([_result("a", findings=(_finding(UNVERIFIED_RESCHEDULE),))], VERSION)
    )

    assert "observation: UNVERIFIED_RESCHEDULE (non-blocking)" in text
    assert "finding: UNVERIFIED_RESCHEDULE" not in text


# --- json ------------------------------------------------------------------


def test_the_json_is_valid_json() -> None:
    assert json.loads(as_json(score([_result("a")], VERSION)))["version"] == VERSION


def test_the_json_carries_the_headline_numbers() -> None:
    payload = to_dict(score([_result("a"), _result("b")], VERSION))

    assert payload["scenarios"] == 2
    assert payload["passed"] == 2
    assert payload["task_success"] == {
        "numerator": 2,
        "denominator": 2,
        "percent": 100.0,
    }


def test_the_json_carries_the_honesty_note() -> None:
    assert to_dict(score([], VERSION))["note"] == HONESTY


def test_the_json_reports_an_undefined_ratio_as_null_rather_than_zero() -> None:
    payload = to_dict(score([_result("a")], VERSION))

    assert payload["escalation"]["precision"]["percent"] is None
    assert payload["escalation"]["precision"]["denominator"] == 0


def test_the_json_lists_every_scenario() -> None:
    payload = to_dict(score([_result("a"), _result("b")], VERSION))

    assert [row["scenario"] for row in payload["results"]] == ["a", "b"]


def test_the_json_marks_which_findings_block() -> None:
    payload = to_dict(
        score([_result("a", findings=(_finding(UNVERIFIED_RESCHEDULE),))], VERSION)
    )

    assert payload["results"][0]["findings"][0]["blocking"] is False


def test_the_json_reports_cost_as_a_string_or_null() -> None:
    """A decimal amount must not go through a float on its way out."""
    priced = to_dict(score([_result("a", cost=Decimal("0.001234"))], VERSION))
    unpriced = to_dict(score([_result("b")], VERSION))

    assert priced["results"][0]["total_cost_usd"] == "0.001234"
    assert unpriced["results"][0]["total_cost_usd"] is None


def test_the_json_is_stable_across_two_renderings() -> None:
    report = score([_result("a")], VERSION)

    assert as_json(report) == as_json(report)
