"""The scenario model: normalisation, and refusing a scenario that cannot be scored.

Every failure here is a dataset fault caught at load time. That matters: a
malformed expectation discovered while a call is running would look exactly
like the receptionist misbehaving.
"""

from datetime import UTC, datetime, time

import pytest

from app.evals.model import say, use_tools
from app.evals.scenario import (
    AppointmentSpec,
    Beat,
    Claim,
    ExpectedTool,
    Expectation,
    HoursSpec,
    Scenario,
    ScenarioError,
    SeedAppointment,
    ServiceSpec,
    World,
    arguments_match,
    normalise_argument,
    normalise_instant,
    placeholder_ref,
    slot_key,
    validate,
    validate_all,
    weekdays,
)

MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"


def _scenario(**overrides) -> Scenario:
    fields = {
        "name": "example",
        "kind": "text",
        "description": "An example.",
        "world": World(services=(ServiceSpec("Haircut"),), hours=weekdays()),
        "script": (Beat(caller="Hello", responses=(say("Hello."),)),),
        "expect": Expectation(task="decline"),
    }
    fields.update(overrides)
    return Scenario(**fields)


# --- normalising time ------------------------------------------------------


def test_an_instant_is_read_into_utc() -> None:
    assert normalise_instant(TEN) == datetime(2026, 3, 2, 10, tzinfo=UTC)


def test_two_spellings_of_one_moment_are_equal() -> None:
    """The same instant written in two offsets is the same instant."""
    assert normalise_instant("2026-03-02T10:00:00+00:00") == normalise_instant(
        "2026-03-02T11:00:00+01:00"
    )


def test_a_naive_timestamp_is_read_in_the_evaluation_timezone() -> None:
    assert normalise_instant("2026-03-02T10:00:00") == datetime(
        2026, 3, 2, 10, tzinfo=UTC
    )


def test_a_datetime_is_accepted_as_well_as_a_string() -> None:
    assert normalise_instant(datetime(2026, 3, 2, 10, tzinfo=UTC)) == normalise_instant(
        TEN
    )


def test_a_slot_key_is_case_folded_and_instant_based() -> None:
    assert slot_key("HAIRCUT", TEN) == slot_key("haircut", "2026-03-02T11:00:00+01:00")


def test_a_slot_key_needs_a_service() -> None:
    with pytest.raises(ScenarioError, match="service name"):
        slot_key("  ", TEN)


# --- normalising arguments -------------------------------------------------


def test_a_time_argument_compares_as_an_instant() -> None:
    assert normalise_argument("2026-03-02T10:00:00+00:00") == normalise_argument(
        "2026-03-02T11:00:00+01:00"
    )


def test_a_text_argument_compares_case_insensitively() -> None:
    assert normalise_argument("Haircut") == normalise_argument("haircut")


def test_a_non_string_argument_is_left_alone() -> None:
    assert normalise_argument(7) == 7
    assert normalise_argument(None) is None


def test_arguments_match_on_a_subset() -> None:
    """A scenario asserts what it cares about and stays silent about the rest."""
    assert arguments_match(
        {"service_name": "Haircut"},
        {"service_name": "haircut", "day": MONDAY, "extra": 1},
    )


def test_arguments_do_not_match_when_a_key_is_missing() -> None:
    assert not arguments_match({"day": MONDAY}, {"service_name": "Haircut"})


def test_arguments_do_not_match_when_a_value_differs() -> None:
    assert not arguments_match({"day": MONDAY}, {"day": "2026-03-03"})


def test_an_empty_expectation_matches_anything() -> None:
    assert arguments_match({}, {"anything": "at all"})


# --- placeholders ----------------------------------------------------------


def test_a_placeholder_is_recognised() -> None:
    assert placeholder_ref("{{appointment:existing}}") == "existing"


def test_whitespace_around_a_placeholder_is_ignored() -> None:
    assert placeholder_ref("  {{appointment:existing}}  ") == "existing"


def test_something_that_is_not_a_placeholder_is_not_one() -> None:
    assert placeholder_ref("existing") is None
    assert placeholder_ref(TEN) is None
    assert placeholder_ref(7) is None


# --- validation ------------------------------------------------------------


def test_a_well_formed_scenario_validates() -> None:
    validate(_scenario())


def test_a_scenario_needs_a_name() -> None:
    with pytest.raises(ScenarioError, match="needs a name"):
        validate(_scenario(name=""))


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ScenarioError, match="not one of"):
        validate(_scenario(kind="carrier-pigeon"))


def test_an_unknown_task_is_refused() -> None:
    with pytest.raises(ScenarioError, match="not one of"):
        validate(_scenario(expect=Expectation(task="vibes")))


def test_a_world_needs_a_service() -> None:
    with pytest.raises(ScenarioError, match="at least one service"):
        validate(_scenario(world=World(services=(), hours=weekdays())))


def test_business_hours_must_close_after_they_open() -> None:
    world = World(
        services=(ServiceSpec("Haircut"),),
        hours=(HoursSpec(0, time(17), time(9)),),
    )
    with pytest.raises(ScenarioError, match="close after"):
        validate(_scenario(world=world))


def test_a_weekday_outside_the_week_is_refused() -> None:
    world = World(services=(ServiceSpec("Haircut"),), hours=(HoursSpec(9),))
    with pytest.raises(ScenarioError, match="not 0"):
        validate(_scenario(world=world))


def test_two_seeded_appointments_may_not_share_a_ref() -> None:
    world = World(
        services=(ServiceSpec("Haircut"),),
        hours=weekdays(),
        appointments=(
            SeedAppointment("same", "Haircut", "Ada", "+447700900123", TEN),
            SeedAppointment("same", "Haircut", "Bea", "+447700900456", TEN),
        ),
    )
    with pytest.raises(ScenarioError, match="share a ref"):
        validate(_scenario(world=world))


def test_an_unreadable_seeded_time_is_refused() -> None:
    world = World(
        services=(ServiceSpec("Haircut"),),
        hours=weekdays(),
        appointments=(
            SeedAppointment("x", "Haircut", "Ada", "+447700900123", "soon"),
        ),
    )
    with pytest.raises(ScenarioError, match="unreadable time"):
        validate(_scenario(world=world))


def test_an_expected_tool_must_be_a_real_tool() -> None:
    expect = Expectation(task="book", expected_tools=(ExpectedTool("do_magic"),))
    with pytest.raises(ScenarioError, match="not a tool"):
        validate(_scenario(expect=expect))


def test_a_forbidden_tool_must_be_a_real_tool() -> None:
    expect = Expectation(task="book", forbidden_tools=frozenset({"do_magic"}))
    with pytest.raises(ScenarioError, match="not a tool"):
        validate(_scenario(expect=expect))


def test_a_tool_cannot_be_both_expected_and_forbidden() -> None:
    expect = Expectation(
        task="book",
        expected_tools=(ExpectedTool("cancel"),),
        forbidden_tools=frozenset({"cancel"}),
    )
    with pytest.raises(ScenarioError, match="both expected and forbidden"):
        validate(_scenario(expect=expect))


def test_a_claim_must_name_a_turn_the_script_has() -> None:
    expect = Expectation(
        task="book", availability_claims=(Claim("Haircut", TEN, 5),)
    )
    with pytest.raises(ScenarioError, match="names turn"):
        validate(_scenario(expect=expect))


def test_a_claim_needs_a_service() -> None:
    expect = Expectation(task="book", availability_claims=(Claim("  ", TEN, 0),))
    with pytest.raises(ScenarioError, match="needs a service"):
        validate(_scenario(expect=expect))


def test_a_claim_with_an_unreadable_time_is_refused() -> None:
    expect = Expectation(
        task="book", availability_claims=(Claim("Haircut", "sometime", 0),)
    )
    with pytest.raises(ScenarioError, match="unreadable time"):
        validate(_scenario(expect=expect))


def test_an_expected_appointment_needs_a_real_status() -> None:
    expect = Expectation(
        task="book",
        final_appointments=(AppointmentSpec("Haircut", "Ada", TEN, "pending"),),
    )
    with pytest.raises(ScenarioError, match="not an appointment status"):
        validate(_scenario(expect=expect))


def test_a_placeholder_must_name_a_seeded_appointment() -> None:
    """A script that refers to a row nobody seeded is a dataset fault."""
    script = (
        Beat(
            caller="Cancel it",
            responses=(
                use_tools(("cancel", {"appointment_id": "{{appointment:ghost}}"})),
            ),
        ),
    )
    with pytest.raises(ScenarioError, match="which the world does not seed"):
        validate(_scenario(script=script))


def test_max_turns_must_be_positive() -> None:
    with pytest.raises(ScenarioError, match="max_turns"):
        validate(_scenario(max_turns=0))


# --- the whole dataset -----------------------------------------------------


def test_two_scenarios_may_not_share_a_name() -> None:
    with pytest.raises(ScenarioError, match="Two scenarios"):
        validate_all([_scenario(), _scenario()])


def test_a_dataset_of_the_wrong_size_is_refused() -> None:
    with pytest.raises(ScenarioError, match="were expected"):
        validate_all([_scenario()], expected=18)


def test_a_dataset_of_the_right_size_passes() -> None:
    validate_all([_scenario(name="one"), _scenario(name="two")], expected=2)


# --- the script ------------------------------------------------------------


def test_a_scenario_flattens_its_responses_in_order() -> None:
    scenario = _scenario(
        script=(
            Beat(caller="a", responses=(say("one"), say("two"))),
            Beat(caller="b", responses=(say("three"),)),
        )
    )

    assert [response.text for response in scenario.responses] == [
        "one",
        "two",
        "three",
    ]
