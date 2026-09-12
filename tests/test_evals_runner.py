"""The runner, against a real database, real tools and a real calendar.

Only the model is scripted here. These tests exist to prove that: that a
booking recorded by the suite is a row PostgreSQL actually holds, that the
exclusion constraint is the thing refusing a double booking, and that the
executor's guard is the thing refusing an unverified one.

They run against the evaluation database, which the runner creates, migrates
and truncates between scenarios — never the test database and never the
development one.
"""

from decimal import Decimal

import pytest
from sqlalchemy import Engine, create_engine, select

from app.config import Settings, get_settings
from app.evals.database import EvalDatabaseError, resolve_eval_database_url
from app.evals.model import say, use_tools
from app.evals.runner import SCRIPTED_PROVIDER, EvalRunner, NoSpeechUsage
from app.evals.scenario import (
    AppointmentSpec,
    Beat,
    Expectation,
    Scenario,
    SeedAppointment,
    ServiceSpec,
    World,
    weekdays,
)
from app.models import Appointment, CallCost, Turn

MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"
ELEVEN = "2026-03-02T11:00:00+00:00"
HAIRCUT = "Haircut"
ADA = ("Ada Lovelace", "+447700900123")
BEA = ("Bea Bramble", "+447700900456")


@pytest.fixture(scope="module")
def runner(database_url: str):
    """One migrated evaluation database for the whole module.

    Module-scoped because migrating is the slow part and every scenario
    truncates before it runs anyway.
    """
    import sqlalchemy

    url = resolve_eval_database_url(get_settings())
    probe = create_engine(url.rsplit("/", 1)[0] + "/postgres")
    try:
        with probe.connect():
            pass
    except sqlalchemy.exc.OperationalError as exc:
        pytest.skip(f"no PostgreSQL for the evaluation database: {exc}")
    finally:
        probe.dispose()

    with EvalRunner() as built:
        yield built


@pytest.fixture
def engine(runner) -> Engine:
    return create_engine(runner.url)


def _world(*seeds: SeedAppointment, services=None) -> World:
    return World(
        services=services or (ServiceSpec(HAIRCUT, 30, "sam"),),
        hours=weekdays(),
        appointments=seeds,
    )


def _check():
    return ("check_availability", {"service_name": HAIRCUT, "day": MONDAY})


def _book(starts_at: str, who=ADA):
    return (
        "book_appointment",
        {
            "service_name": HAIRCUT,
            "starts_at": starts_at,
            "customer_name": who[0],
            "phone": who[1],
        },
    )


def _scenario(script, *, world=None, expect=None, **overrides) -> Scenario:
    fields = {
        "name": "under_test",
        "kind": "text",
        "description": "A scenario built inside a test.",
        "world": world or _world(),
        "script": script,
        "expect": expect or Expectation(task="decline"),
    }
    fields.update(overrides)
    return Scenario(**fields)


# --- the real stack --------------------------------------------------------


def test_a_booking_reaches_the_database(runner, engine: Engine) -> None:
    scenario = _scenario(
        (
            Beat("Monday please", (use_tools(_check()), say("I have 10:00."))),
            Beat("10:00 then", (use_tools(_book(TEN)), say("Booked."))),
        )
    )

    runner.run(scenario)

    with engine.connect() as connection:
        rows = connection.execute(select(Appointment.customer_name)).scalars().all()
    assert rows == [ADA[0]]


def test_the_trace_reports_what_the_database_holds(runner) -> None:
    scenario = _scenario(
        (
            Beat("Monday please", (use_tools(_check()), say("I have 10:00."))),
            Beat("10:00 then", (use_tools(_book(TEN)), say("Booked."))),
        )
    )

    trace = runner.run(scenario)

    assert trace.final_appointments == [AppointmentSpec(HAIRCUT, ADA[0], TEN)]


def test_the_transcript_is_written_by_the_dialogue_layer(runner, engine: Engine) -> None:
    scenario = _scenario((Beat("Hello", (say("Hello there."),)),))

    runner.run(scenario)

    with engine.connect() as connection:
        turns = connection.execute(select(Turn.text)).scalars().all()
    assert turns == ["Hello", "Hello there."]


def test_the_executors_guard_refuses_an_unverified_booking(runner) -> None:
    """No availability check at all: the real guard, not a check of our own."""
    scenario = _scenario((Beat("10:00 please", (use_tools(_book(TEN)), say("No."))),))

    trace = runner.run(scenario)

    assert trace.tool_calls[0].success is False
    assert trace.tool_calls[0].data.get("unverified_slot") is True
    assert trace.final_appointments == []


def test_the_exclusion_constraint_refuses_the_second_booking(runner) -> None:
    """Two callers, one slot, both verified. PostgreSQL decides."""
    scenario = _scenario(
        (
            Beat("Monday", (use_tools(_check()), say("I have 10:00."))),
            Beat("Ada at 10:00", (use_tools(_book(TEN, ADA)), say("Booked."))),
            Beat("Bea at 10:00", (use_tools(_book(TEN, BEA)), say("Sorry."))),
        )
    )

    trace = runner.run(scenario)

    second = [r for r in trace.tool_calls if r.tool_name == "book_appointment"][1]
    assert second.success is False
    assert second.data.get("slot_taken") is True
    assert len(trace.final_appointments) == 1


def test_business_hours_really_constrain_availability(runner) -> None:
    """A Saturday is closed, so the calendar offers nothing."""
    scenario = _scenario(
        (
            Beat(
                "Saturday?",
                (
                    use_tools(
                        ("check_availability", {"service_name": HAIRCUT, "day": "2026-03-07"})
                    ),
                    say("Nothing then, I'm afraid."),
                ),
            ),
        )
    )

    trace = runner.run(scenario)

    assert trace.tool_calls[0].success is True
    assert trace.tool_calls[0].data["slots"] == []


# --- the ledger ------------------------------------------------------------


def test_the_ledger_is_reconstructed_from_the_trace(runner) -> None:
    scenario = _scenario(
        (Beat("Monday", (use_tools(_check()), say("Plenty free."))),)
    )

    trace = runner.run(scenario)

    offered = {instant.isoformat() for _, instant in trace.offered}
    assert "2026-03-02T09:00:00+00:00" in offered
    assert "2026-03-02T10:00:00+00:00" in offered


def test_the_ledger_is_empty_before_anything_is_checked(runner) -> None:
    trace = runner.run(_scenario((Beat("Hello", (say("Hello."),)),)))

    assert trace.offered == set()


def test_a_seeded_appointment_is_not_offered(runner) -> None:
    world = _world(SeedAppointment("taken", HAIRCUT, BEA[0], BEA[1], TEN))
    scenario = _scenario(
        (Beat("Monday", (use_tools(_check()), say("Some free."))),), world=world
    )

    trace = runner.run(scenario)

    offered = {instant.isoformat() for _, instant in trace.offered}
    assert "2026-03-02T10:00:00+00:00" not in offered


# --- isolation between scenarios -------------------------------------------


def test_each_scenario_starts_from_an_empty_database(runner, engine: Engine) -> None:
    booked = _scenario(
        (
            Beat("Monday", (use_tools(_check()), say("I have 10:00."))),
            Beat("10:00", (use_tools(_book(TEN)), say("Booked."))),
        )
    )
    runner.run(booked)

    trace = runner.run(_scenario((Beat("Hello", (say("Hello."),)),)))

    assert trace.final_appointments == []
    with engine.connect() as connection:
        assert connection.execute(select(Appointment)).all() == []


def test_a_seeded_world_is_rebuilt_each_time(runner) -> None:
    world = _world(SeedAppointment("existing", HAIRCUT, ADA[0], ADA[1], TEN))
    scenario = _scenario((Beat("Hello", (say("Hello."),)),), world=world)

    first = runner.run(scenario)
    second = runner.run(scenario)

    assert len(first.final_appointments) == 1
    assert len(second.final_appointments) == 1


# --- placeholders ----------------------------------------------------------


def test_a_placeholder_becomes_the_real_identifier(runner) -> None:
    world = _world(SeedAppointment("existing", HAIRCUT, ADA[0], ADA[1], TEN))
    scenario = _scenario(
        (
            Beat(
                "Cancel it",
                (
                    use_tools(("cancel", {"appointment_id": "{{appointment:existing}}"})),
                    say("Cancelled."),
                ),
            ),
        ),
        world=world,
    )

    trace = runner.run(scenario)

    assert trace.tool_calls[0].success is True
    assert trace.tool_calls[0].arguments["appointment_id"] == trace.appointment_refs[
        "existing"
    ]
    assert trace.final_appointments[0].status == "cancelled"


def test_the_trace_records_what_each_reference_became(runner) -> None:
    world = _world(SeedAppointment("existing", HAIRCUT, ADA[0], ADA[1], TEN))
    trace = runner.run(_scenario((Beat("Hi", (say("Hi."),)),), world=world))

    assert set(trace.appointment_refs) == {"existing"}


# --- failures --------------------------------------------------------------


def test_running_out_of_script_is_recorded_as_an_authoring_fault(runner) -> None:
    scenario = _scenario((Beat("Hello", ()),))

    trace = runner.run(scenario)

    assert trace.error_category == "SCRIPT_EXHAUSTED"
    assert "ran out" in (trace.error or "")


def test_a_scenario_that_fails_does_not_stop_the_next_one(runner) -> None:
    runner.run(_scenario((Beat("Hello", ()),)))

    trace = runner.run(_scenario((Beat("Hello", (say("Hello."),)),)))

    assert trace.error_category is None


def test_a_fixture_the_calendar_would_refuse_is_a_dataset_fault(runner) -> None:
    """A world the application could not have produced is not a world."""
    world = _world(
        SeedAppointment("closed", HAIRCUT, ADA[0], ADA[1], "2026-03-07T10:00:00+00:00")
    )

    trace = runner.run(_scenario((Beat("Hi", (say("Hi."),)),), world=world))

    assert trace.error_category == "SCRIPT_EXHAUSTED"
    assert "calendar refused" in (trace.error or "")


# --- cost, from milestone 8 ------------------------------------------------


def test_a_model_turn_records_its_usage(runner, engine: Engine) -> None:
    runner.run(_scenario((Beat("Hello", (say("Hello."),)),)))

    with engine.connect() as connection:
        rows = connection.execute(
            select(CallCost.component, CallCost.provider, CallCost.input_units)
        ).all()

    assert [(str(component), provider) for component, provider, _ in rows] == [
        ("llm", SCRIPTED_PROVIDER)
    ]


def test_a_text_evaluation_buys_no_speech(runner, engine: Engine) -> None:
    """No provider was asked, so no row claims one charged nothing."""
    runner.run(_scenario((Beat("Hello", (say("Hello."),)),)))

    with engine.connect() as connection:
        components = {
            str(component)
            for component in connection.execute(select(CallCost.component)).scalars()
        }

    assert components == {"llm"}


def test_the_cost_provider_does_not_name_a_vendor(runner) -> None:
    """Nothing reading these rows should think a real model was billed."""
    assert SCRIPTED_PROVIDER == "scripted"


def test_no_price_configured_leaves_the_call_unpriced(runner) -> None:
    trace = runner.run(_scenario((Beat("Hello", (say("Hello."),)),)))

    assert trace.total_cost_usd is None
    assert trace.component_costs == {"llm": None}


def test_a_configured_price_produces_a_measured_cost(database_url: str) -> None:
    """Milestone 8 unchanged: the price comes from configuration, as ever."""
    priced = get_settings().model_copy(
        update={
            "llm_input_usd_per_mtok": "1",
            "llm_output_usd_per_mtok": "4",
        }
    )
    with EvalRunner(priced) as runner:
        trace = runner.run(_scenario((Beat("Hello", (say("Hello."),)),)))

    # 400 input at 0.000001 plus 40 output at 0.000004.
    assert trace.total_cost_usd == Decimal("0.000560")


def test_no_speech_usage_reports_empty_providers() -> None:
    usage = NoSpeechUsage()

    assert usage.stt_provider == ""
    assert usage.tts_provider == ""
    assert usage.stt_audio_ms is None


# --- the runner's own guards -----------------------------------------------


def test_the_runner_refuses_a_database_it_may_not_touch() -> None:
    settings = Settings(
        _env_file=None,
        eval_database_url="postgresql+psycopg://u:p@localhost:5432/voicedesk",
    )

    with pytest.raises(EvalDatabaseError, match="Refusing"):
        EvalRunner(settings)


def test_the_runner_fixes_the_timezone(runner) -> None:
    """A dataset of fixed dates has to mean the same thing everywhere."""
    assert runner.settings.business_timezone == "UTC"


def test_the_runner_turns_cost_tracking_on(runner) -> None:
    assert runner.settings.cost_tracking_enabled is True


def test_the_runner_points_at_the_evaluation_database(runner) -> None:
    assert runner.settings.database_url == runner.url
    assert runner.url.endswith("_evals")


def test_the_runner_must_be_used_as_a_context_manager() -> None:
    with pytest.raises(RuntimeError, match="context manager"):
        EvalRunner().run(_scenario((Beat("Hello", (say("Hello."),)),)))
