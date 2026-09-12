"""The dataset: eighteen scenarios, all of them valid and none of them real.

Two rules matter most here. The count is asserted, so losing a scenario is a
loud failure rather than a quietly smaller suite. And no real personal
information appears: the names are invented and the numbers are from the
reserved drama range, which is never allocated to a subscriber.
"""

import pathlib
import re

import pytest

from app.evals import dataset
from app.evals.scenario import ScenarioError, validate_all
from app.tools import TOOLS

DATASET = pathlib.Path(__file__).resolve().parent.parent / "app" / "evals" / "dataset"


# --- shape -----------------------------------------------------------------


def test_the_dataset_holds_exactly_eighteen_scenarios() -> None:
    assert len(dataset.all_scenarios()) == dataset.EXPECTED_SCENARIOS == 18


def test_the_specifications_minimum_is_met() -> None:
    """The specification asks for at least fifteen written call scenarios."""
    assert len(dataset.SCENARIOS) >= 15


def test_every_scenario_validates() -> None:
    validate_all(dataset.SCENARIOS, expected=dataset.EXPECTED_SCENARIOS)


def test_every_name_is_unique() -> None:
    names = dataset.names()

    assert len(set(names)) == len(names)


def test_the_approved_scenarios_are_all_present() -> None:
    assert set(dataset.names()) == {
        "booking_available",
        "booking_caller_changes_mind",
        "booking_unavailable_slot",
        "booking_slot_taken_between_check_and_book",
        "booking_unknown_service",
        "booking_ambiguous_service",
        "booking_missing_details",
        "reschedule_success",
        "reschedule_conflict",
        "cancel_success",
        "cancel_already_cancelled",
        "take_message",
        "escalate_explicit_request",
        "escalate_medical",
        "escalate_legal",
        "escalate_complaint_refund",
        "realtime_barge_in_booking_survives",
        "realtime_silence_asks_no_model",
    }


def test_every_family_is_represented() -> None:
    dataset.all_scenarios()  # validates families as well as scenarios

    assert set(dataset.FAMILIES) >= {
        "booking",
        "unavailable",
        "unknown_service",
        "ambiguity",
        "missing_details",
        "reschedule",
        "cancel",
        "message",
        "escalation",
        "barge_in",
        "silence",
    }


def test_every_scenario_belongs_to_a_family() -> None:
    covered = {name for members in dataset.FAMILIES.values() for name in members}

    assert set(dataset.names()) == covered


def test_both_kinds_are_present() -> None:
    kinds = {scenario.kind for scenario in dataset.SCENARIOS}

    assert kinds == {"text", "realtime"}


def test_there_are_two_realtime_scenarios() -> None:
    realtime = [s for s in dataset.SCENARIOS if s.kind == "realtime"]

    assert len(realtime) == 2


def test_every_scenario_describes_itself() -> None:
    for scenario in dataset.SCENARIOS:
        assert len(scenario.description) > 20, scenario.name


# --- what the dataset covers -----------------------------------------------


def test_four_scenarios_require_an_escalation() -> None:
    """The specification asks for correct escalation on all four triggers."""
    escalating = [s for s in dataset.SCENARIOS if s.expect.should_escalate]

    assert len(escalating) == 4


def test_no_other_scenario_expects_an_escalation() -> None:
    for scenario in dataset.SCENARIOS:
        if scenario.name.startswith("escalate_"):
            assert scenario.expect.should_escalate is True, scenario.name
        else:
            assert scenario.expect.should_escalate is False, scenario.name


def test_the_dataset_is_adversarial_as_well_as_happy() -> None:
    """Scenarios that expect a tool to fail, because refusal is the behaviour."""
    adversarial = [s for s in dataset.SCENARIOS if s.expect.allow_tool_failures]

    assert len(adversarial) >= 4


def test_some_scenario_forbids_booking() -> None:
    forbidding = [
        s for s in dataset.SCENARIOS if "book_appointment" in s.expect.forbidden_tools
    ]

    assert forbidding


def test_every_expected_tool_is_a_real_tool() -> None:
    for scenario in dataset.SCENARIOS:
        for expected in scenario.expect.expected_tools:
            assert expected.name in TOOLS, f"{scenario.name}: {expected.name}"


def test_every_tool_is_exercised_somewhere() -> None:
    """All six, or the suite is not evaluating the whole tool layer."""
    named = {
        expected.name
        for scenario in dataset.SCENARIOS
        for expected in scenario.expect.expected_tools
    }

    assert named == set(TOOLS)


# --- selection -------------------------------------------------------------


def test_scenarios_can_be_selected_by_name() -> None:
    selected = dataset.by_name(("take_message", "cancel_success"))

    assert [s.name for s in selected] == ["cancel_success", "take_message"]


def test_selection_keeps_dataset_order() -> None:
    """So two runs of one selection produce the same report."""
    selected = dataset.by_name(("realtime_silence_asks_no_model", "booking_available"))

    assert [s.name for s in selected] == [
        "booking_available",
        "realtime_silence_asks_no_model",
    ]


def test_an_unknown_name_is_refused_with_the_list() -> None:
    with pytest.raises(ScenarioError, match="booking_available"):
        dataset.by_name(("nonsense",))


# --- no real people --------------------------------------------------------

# Ofcom reserves 07700 900000–900999 for drama. A number outside it in this
# dataset would be somebody's.
DRAMA = re.compile(r"\+447700900\d{3}")
PHONE_SHAPED = re.compile(r"\+\d{9,15}")


def _sources() -> list[pathlib.Path]:
    return sorted(DATASET.rglob("*.py"))


def test_every_telephone_number_is_from_the_reserved_range() -> None:
    for path in _sources():
        for found in PHONE_SHAPED.findall(path.read_text()):
            assert DRAMA.fullmatch(found), f"{path.name}: {found}"


def test_the_dataset_names_no_real_looking_email_address() -> None:
    for path in _sources():
        assert "@" not in path.read_text().replace("@dataclass", ""), path.name


def test_the_dataset_holds_no_credential() -> None:
    for path in _sources():
        lowered = path.read_text().lower()
        for word in ("api_key", "secret", "password", "token="):
            assert word not in lowered, f"{path.name} mentions {word}"


# --- fixed in time ---------------------------------------------------------


def test_the_dataset_reads_no_clock() -> None:
    """Fixed dates only: availability is a pure function of the day asked for."""
    for path in _sources():
        source = path.read_text()
        for forbidden in ("datetime.now", "utcnow", "date.today", "time.time"):
            assert forbidden not in source, f"{path.name} uses {forbidden}"


def test_every_date_in_the_dataset_is_in_the_fixed_week() -> None:
    dates = set()
    for path in _sources():
        dates |= set(re.findall(r"\d{4}-\d{2}-\d{2}", path.read_text()))

    assert dates <= {"2026-03-02", "2026-03-03", "2026-03-07"}
