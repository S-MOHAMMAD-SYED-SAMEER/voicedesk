"""`python -m app.evals`: what it prints, what it writes, and what it exits with.

The whole suite runs here, against the real evaluation database. That makes
this file the one that answers the question the milestone exists to answer —
what does VoiceDesk actually score — so it asserts the numbers rather than
merely that something was printed.
"""

import json

import pytest
import sqlalchemy
from sqlalchemy import create_engine

from app.config import get_settings
from app.evals import EVAL_SUITE_VERSION, dataset
from app.evals.__main__ import main, run
from app.evals.database import resolve_eval_database_url


@pytest.fixture(scope="module")
def evaluated(database_url: str):
    """The whole suite, run once, for every test in this file to read."""
    url = resolve_eval_database_url(get_settings())
    probe = create_engine(url.rsplit("/", 1)[0] + "/postgres")
    try:
        with probe.connect():
            pass
    except sqlalchemy.exc.OperationalError as exc:
        pytest.skip(f"no PostgreSQL for the evaluation database: {exc}")
    finally:
        probe.dispose()

    return run(dataset.all_scenarios())


# --- listing ---------------------------------------------------------------


def test_list_prints_every_scenario_and_runs_nothing(capsys) -> None:
    assert main(["--list"]) == 0

    printed = capsys.readouterr().out
    for name in dataset.names():
        assert name in printed
    assert "Task success" not in printed


def test_list_names_each_kind(capsys) -> None:
    main(["--list"])

    printed = capsys.readouterr().out
    assert "\ttext\t" in printed
    assert "\trealtime\t" in printed


def test_an_unknown_scenario_cannot_run(capsys) -> None:
    assert main(["--scenario", "nonsense"]) == 2
    assert "No scenario called nonsense" in capsys.readouterr().err


# --- running ---------------------------------------------------------------


def test_the_whole_suite_runs(evaluated) -> None:
    assert evaluated.total == 18


def test_the_suite_reports_the_version_it_ran_under(evaluated) -> None:
    assert evaluated.version == EVAL_SUITE_VERSION == "m9-v1"


def test_no_availability_is_ever_hallucinated(evaluated) -> None:
    """The specification's one absolute: this must be zero."""
    assert evaluated.hallucinated_availability == 0


def test_every_tool_call_the_suite_expected_was_made_correctly(evaluated) -> None:
    assert evaluated.tools.wrong == 0
    assert evaluated.tools.missing == 0
    assert evaluated.tools.duplicate == 0
    assert evaluated.tools.invalid_argument == 0


def test_escalation_is_exact_across_the_suite(evaluated) -> None:
    assert evaluated.escalation_precision.numerator == 4
    assert evaluated.escalation_precision.denominator == 4
    assert evaluated.escalation_recall.numerator == 4
    assert evaluated.escalation_recall.denominator == 4


def test_no_scenario_failed_because_of_its_own_script(evaluated) -> None:
    """A dataset fault would make every other number meaningless."""
    assert evaluated.authoring_failures == 0


def test_the_suite_surfaces_the_unguarded_reschedule(evaluated) -> None:
    """Measured, reported, and deliberately not counted against success."""
    assert evaluated.unverified_reschedules == 1
    assert any(
        result.name == "reschedule_success" and result.passed
        for result in evaluated.results
    )


def test_the_reschedule_conflict_now_completes(evaluated) -> None:
    """The defect this scenario found in milestone 9 is fixed.

    A conflicting reschedule used to leave the call's session in a
    pending-rollback state, so the turn could not be written and the caller
    heard nothing. `CalendarService._write` now rolls back before raising, and
    the scenario passes with its ground truth **unchanged** — the reschedule
    still fails, both appointments still stand, and the caller is still told.
    """
    result = next(r for r in evaluated.results if r.name == "reschedule_conflict")

    assert result.passed is True
    assert result.assessment.findings == []
    assert result.trace.error is None
    # The move was still refused, and nothing moved.
    assert [record.success for record in result.trace.tool_calls] == [False]
    assert len(result.trace.final_appointments) == 2


def test_every_scenario_passes(evaluated) -> None:
    failed = [r.name for r in evaluated.results if not r.passed]

    assert failed == []


def test_task_success_is_what_it_is(evaluated) -> None:
    """Eighteen of eighteen, with no scenario or expectation changed."""
    assert evaluated.passed == 18
    assert evaluated.task_success.percent == pytest.approx(100.0, abs=0.1)


def test_no_provider_was_ever_called(evaluated) -> None:
    """Every cost row names the scripted model, never a vendor."""
    for result in evaluated.results:
        for component in result.trace.component_costs:
            assert component in ("llm", "stt", "tts", "telephony")
    assert evaluated.measured_calls + evaluated.unpriced_calls == 18


def test_with_no_prices_configured_every_call_is_unpriced(evaluated) -> None:
    assert evaluated.unpriced_calls == 18
    assert evaluated.total_cost_usd is None


# --- selection -------------------------------------------------------------


def test_one_scenario_can_be_run_alone(database_url: str, capsys) -> None:
    code = main(["--scenario", "take_message", "--quiet"])

    assert code == 0
    assert "Scenarios: 1" in capsys.readouterr().out


def test_scenarios_can_be_selected_repeatedly(database_url: str, capsys) -> None:
    code = main(
        ["--scenario", "take_message", "--scenario", "cancel_success", "--quiet"]
    )

    assert code == 0
    assert "Scenarios: 2" in capsys.readouterr().out


# --- output ----------------------------------------------------------------


def test_quiet_prints_the_headline_without_the_detail(
    database_url: str, capsys
) -> None:
    main(["--scenario", "take_message", "--quiet"])

    printed = capsys.readouterr().out
    assert "Task success" in printed
    assert "PASS take_message" not in printed


def test_the_full_report_includes_the_detail(database_url: str, capsys) -> None:
    main(["--scenario", "take_message"])

    printed = capsys.readouterr().out
    assert "PASS take_message" in printed
    assert "no real model's conversational quality" in printed


def test_json_is_written_where_it_was_asked_for(
    database_url: str, tmp_path, capsys
) -> None:
    target = tmp_path / "report.json"

    main(["--scenario", "take_message", "--json", str(target), "--quiet"])

    payload = json.loads(target.read_text())
    assert payload["version"] == EVAL_SUITE_VERSION
    assert payload["results"][0]["scenario"] == "take_message"


# --- exit codes ------------------------------------------------------------


def test_a_passing_selection_exits_zero(database_url: str, capsys) -> None:
    assert main(["--scenario", "booking_available", "--quiet"]) == 0
    capsys.readouterr()


def test_a_selection_that_cannot_meet_its_threshold_exits_one(
    database_url: str, capsys
) -> None:
    """Every scenario passes now, so the threshold is what can refuse."""
    assert (
        main(["--scenario", "booking_available", "--fail-under", "101", "--quiet"])
        == 1
    )
    capsys.readouterr()


def test_a_threshold_that_is_met_exits_zero(database_url: str, capsys) -> None:
    assert main(["--scenario", "booking_available", "--fail-under", "100", "--quiet"]) == 0
    capsys.readouterr()


def test_a_threshold_that_is_not_met_exits_one(database_url: str, capsys) -> None:
    assert (
        main(["--scenario", "cancel_success", "--fail-under", "100.5", "--quiet"]) == 1
    )
    capsys.readouterr()


def test_the_release_gate_is_met(database_url: str, capsys) -> None:
    """The gate milestone 10 releases against: every scenario, all of them."""
    code = main(["--fail-under", "100", "--quiet"])

    assert code == 0
    capsys.readouterr()
