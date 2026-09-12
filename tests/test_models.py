"""The models, against a real migrated PostgreSQL database.

Nothing here asserts on an object that was never persisted: the behaviour
being checked — foreign keys, cascades, check constraints and above all the
double-booking constraint — lives in the database, not in Python.
"""

import uuid
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    Appointment,
    AppointmentStatus,
    BusinessHours,
    Call,
    CallCost,
    CallDirection,
    CallOutcome,
    CostComponent,
    Service,
    ToolCall,
    Turn,
    TurnRole,
)

NINE_AM = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)


def _call(session: Session, **overrides: object) -> Call:
    call = Call(
        **{
            "from_number": "+447700900123",
            "to_number": "+441173450000",
            **overrides,
        }
    )
    session.add(call)
    session.commit()
    return call


def _service(session: Session, staff_id: str = "sam", **overrides: object) -> Service:
    service = Service(
        **{
            "name": "Haircut",
            "duration_minutes": 30,
            "staff_id": staff_id,
            **overrides,
        }
    )
    session.add(service)
    session.commit()
    return service


def _appointment(
    session: Session,
    service: Service,
    *,
    starts_at: datetime = NINE_AM,
    minutes: int = 30,
    staff_id: str | None = None,
    **overrides: object,
) -> Appointment:
    appointment = Appointment(
        customer_name="Ada Lovelace",
        phone="+447700900123",
        service_id=service.id,
        staff_id=staff_id or service.staff_id,
        starts_at=starts_at,
        ends_at=starts_at + timedelta(minutes=minutes),
        **overrides,
    )
    session.add(appointment)
    session.commit()
    return appointment


# --- calls, turns, tool calls ---------------------------------------------


def test_a_call_persists_with_its_defaults(session: Session) -> None:
    call = _call(session)

    assert isinstance(call.id, uuid.UUID)
    assert call.direction is CallDirection.INBOUND
    assert call.started_at is not None
    # A call in progress has not ended, has no outcome and has cost nothing yet.
    assert call.ended_at is None
    assert call.outcome is None
    assert call.total_cost_usd is None


def test_a_finished_call_records_its_outcome_and_cost(session: Session) -> None:
    call = _call(session)
    call.ended_at = datetime.now(UTC)
    call.outcome = CallOutcome.BOOKED
    call.total_cost_usd = Decimal("0.184000")
    session.commit()
    call_id = call.id
    session.expunge_all()

    stored = session.get(Call, call_id)
    assert stored.outcome is CallOutcome.BOOKED
    assert stored.total_cost_usd == Decimal("0.184000")


def test_a_call_requires_both_numbers(session: Session) -> None:
    with pytest.raises(IntegrityError):
        session.execute(
            text("insert into calls (to_number) values ('+441173450000')")
        )
    session.rollback()


def test_turns_belong_to_a_call_and_load_in_order(session: Session) -> None:
    call = _call(session)
    session.add_all(
        [
            Turn(call_id=call.id, role=TurnRole.CALLER, text="Hi, can I book a cut?"),
            Turn(call_id=call.id, role=TurnRole.AGENT, text="Of course — when for?"),
        ]
    )
    session.commit()
    call_id = call.id
    session.expunge_all()

    stored = session.get(Call, call_id)
    assert [turn.role for turn in stored.turns] == [TurnRole.CALLER, TurnRole.AGENT]


def test_a_turn_records_only_the_latencies_that_apply(session: Session) -> None:
    """A caller turn has an STT latency; it has no LLM or TTS latency."""
    call = _call(session)
    turn = Turn(
        call_id=call.id,
        role=TurnRole.CALLER,
        text="Tuesday morning please",
        audio_ms=2100,
        stt_latency_ms=310,
    )
    session.add(turn)
    session.commit()
    turn_id = turn.id
    session.expunge_all()

    stored = session.get(Turn, turn_id)
    assert stored.stt_latency_ms == 310
    assert stored.llm_latency_ms is None
    assert stored.tts_latency_ms is None


def test_a_turn_needs_a_real_call(session: Session) -> None:
    session.add(Turn(call_id=uuid.uuid4(), role=TurnRole.AGENT, text="orphan"))

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_a_tool_call_records_arguments_and_result(session: Session) -> None:
    call = _call(session)
    turn = Turn(call_id=call.id, role=TurnRole.AGENT, text="Let me check.")
    session.add(turn)
    session.commit()

    tool_call = ToolCall(
        turn_id=turn.id,
        tool_name="check_availability",
        arguments={"service": "Haircut", "date": "2026-03-02"},
        result={"slots": ["09:00", "09:30"]},
        success=True,
        latency_ms=42,
    )
    session.add(tool_call)
    session.commit()
    tool_call_id = tool_call.id
    session.expunge_all()

    stored = session.get(ToolCall, tool_call_id)
    assert stored.arguments["service"] == "Haircut"
    assert stored.result["slots"] == ["09:00", "09:30"]
    assert stored.error is None


def test_a_failed_tool_call_has_an_error_and_no_result(session: Session) -> None:
    call = _call(session)
    turn = Turn(call_id=call.id, role=TurnRole.AGENT, text="One moment.")
    session.add(turn)
    session.commit()

    session.add(
        ToolCall(
            turn_id=turn.id,
            tool_name="book_appointment",
            arguments={"slot": "09:00"},
            success=False,
            error="slot taken",
        )
    )
    session.commit()

    stored = session.execute(select(ToolCall)).scalar_one()
    assert stored.success is False
    assert stored.result is None
    assert stored.error == "slot taken"


def test_deleting_a_call_takes_its_transcript_with_it(session: Session) -> None:
    call = _call(session)
    turn = Turn(call_id=call.id, role=TurnRole.AGENT, text="Hello")
    session.add(turn)
    session.commit()
    session.add(
        ToolCall(turn_id=turn.id, tool_name="take_message", arguments={}, success=True)
    )
    session.commit()

    session.delete(session.get(Call, call.id))
    session.commit()

    assert session.execute(select(Turn)).scalars().all() == []
    assert session.execute(select(ToolCall)).scalars().all() == []


# --- services and business hours -------------------------------------------


def test_a_service_persists_and_defaults_to_active(session: Session) -> None:
    service = _service(session)

    assert service.active is True
    assert service.duration_minutes == 30


def test_a_service_cannot_have_a_non_positive_duration(session: Session) -> None:
    session.add(Service(name="Impossible", duration_minutes=0, staff_id="sam"))

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_business_hours_persist_as_wall_clock_times(session: Session) -> None:
    hours = BusinessHours(weekday=0, opens_at=time(9, 0), closes_at=time(17, 30))
    session.add(hours)
    session.commit()
    hours_id = hours.id
    session.expunge_all()

    stored = session.get(BusinessHours, hours_id)
    assert stored.opens_at == time(9, 0)
    assert stored.closes_at == time(17, 30)


@pytest.mark.parametrize("weekday", [-1, 7])
def test_a_weekday_outside_monday_to_sunday_is_rejected(
    session: Session, weekday: int
) -> None:
    session.add(
        BusinessHours(weekday=weekday, opens_at=time(9), closes_at=time(17))
    )

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_closing_before_opening_is_rejected(session: Session) -> None:
    session.add(BusinessHours(weekday=0, opens_at=time(17), closes_at=time(9)))

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_a_weekday_may_have_several_opening_periods(session: Session) -> None:
    """A lunch break is two rows, so weekday is deliberately not unique."""
    session.add_all(
        [
            BusinessHours(weekday=0, opens_at=time(9), closes_at=time(12, 30)),
            BusinessHours(weekday=0, opens_at=time(13, 30), closes_at=time(17)),
        ]
    )
    session.commit()

    assert len(session.execute(select(BusinessHours)).scalars().all()) == 2


# --- appointments ----------------------------------------------------------


def test_an_appointment_links_its_service_and_call(session: Session) -> None:
    call = _call(session)
    service = _service(session)

    appointment = _appointment(session, service, call_id=call.id)
    appointment_id, expected_call_id = appointment.id, call.id

    session.expunge_all()
    stored = session.get(Appointment, appointment_id)
    assert stored.service.name == "Haircut"
    assert stored.call.id == expected_call_id
    assert stored.status is AppointmentStatus.BOOKED


def test_an_appointment_may_exist_without_a_call(session: Session) -> None:
    service = _service(session)

    appointment = _appointment(session, service)

    assert appointment.call_id is None


def test_deleting_a_call_keeps_its_appointment(session: Session) -> None:
    """The booking outlives the conversation that made it."""
    call = _call(session)
    service = _service(session)
    appointment = _appointment(session, service, call_id=call.id)
    appointment_id = appointment.id

    session.delete(session.get(Call, call.id))
    session.commit()
    session.expunge_all()

    stored = session.get(Appointment, appointment_id)
    assert stored is not None
    assert stored.call_id is None


def test_a_service_with_appointments_cannot_be_deleted(session: Session) -> None:
    service = _service(session)
    _appointment(session, service)

    session.delete(session.get(Service, service.id))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_an_appointment_must_end_after_it_starts(session: Session) -> None:
    service = _service(session)

    session.add(
        Appointment(
            customer_name="Ada",
            phone="+447700900123",
            service_id=service.id,
            staff_id=service.staff_id,
            starts_at=NINE_AM,
            ends_at=NINE_AM - timedelta(minutes=30),
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


# --- the constraint the specification insists on ---------------------------


def test_the_exclusion_constraint_exists_in_the_database(
    migrated_engine: Engine,
) -> None:
    with migrated_engine.connect() as connection:
        definition = connection.execute(
            text(
                "select pg_get_constraintdef(oid) from pg_constraint "
                "where conname = 'ck_appointments_no_double_booking'"
            )
        ).scalar_one()

    assert "EXCLUDE USING gist" in definition
    assert "staff_id WITH =" in definition
    assert "&&" in definition


def test_two_overlapping_bookings_for_one_staff_member_are_impossible(
    session: Session,
) -> None:
    """The database refuses, so no booking code can get this wrong."""
    service = _service(session, staff_id="sam")
    _appointment(session, service, starts_at=NINE_AM, minutes=30)

    with pytest.raises(DBAPIError) as raised:
        _appointment(
            session, service, starts_at=NINE_AM + timedelta(minutes=15), minutes=30
        )
    session.rollback()

    assert "ck_appointments_no_double_booking" in str(raised.value)


def test_a_booking_that_merely_touches_another_is_allowed(session: Session) -> None:
    """09:00–09:30 and 09:30–10:00 do not overlap: the range is half-open."""
    service = _service(session, staff_id="sam")
    _appointment(session, service, starts_at=NINE_AM, minutes=30)

    _appointment(session, service, starts_at=NINE_AM + timedelta(minutes=30))

    assert len(session.execute(select(Appointment)).scalars().all()) == 2


def test_two_staff_may_be_booked_at_the_same_time(session: Session) -> None:
    sam = _service(session, staff_id="sam")
    jo = _service(session, staff_id="jo", name="Colour")

    _appointment(session, sam, starts_at=NINE_AM)
    _appointment(session, jo, starts_at=NINE_AM)

    assert len(session.execute(select(Appointment)).scalars().all()) == 2


def test_a_cancelled_appointment_stops_holding_its_slot(session: Session) -> None:
    service = _service(session, staff_id="sam")
    first = _appointment(session, service, starts_at=NINE_AM)

    first.status = AppointmentStatus.CANCELLED
    session.commit()
    _appointment(session, service, starts_at=NINE_AM)

    booked = session.execute(
        select(Appointment).filter_by(status=AppointmentStatus.BOOKED)
    ).scalars().all()
    assert len(booked) == 1


def test_the_constraint_survives_a_concurrent_insert(
    migrated_engine: Engine, session: Session
) -> None:
    """Two connections, neither aware of the other — one must still lose.

    This is the case application-level checking cannot cover: both callers
    pass an availability check before either has committed.
    """
    service = _service(session, staff_id="sam")

    with Session(migrated_engine) as first, Session(migrated_engine) as second:
        first.add(
            Appointment(
                customer_name="Ada",
                phone="+1",
                service_id=service.id,
                staff_id="sam",
                starts_at=NINE_AM,
                ends_at=NINE_AM + timedelta(minutes=30),
            )
        )
        second.add(
            Appointment(
                customer_name="Bob",
                phone="+2",
                service_id=service.id,
                staff_id="sam",
                starts_at=NINE_AM + timedelta(minutes=10),
                ends_at=NINE_AM + timedelta(minutes=40),
            )
        )
        first.commit()

        with pytest.raises(DBAPIError):
            second.commit()
        second.rollback()

    assert len(session.execute(select(Appointment)).scalars().all()) == 1


# --- schema shape ----------------------------------------------------------


def test_primary_keys_are_uuids(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)

    for table in ("calls", "turns", "tool_calls", "appointments", "services",
                  "business_hours"):
        columns = {column["name"]: column for column in inspector.get_columns(table)}
        assert columns["id"]["type"].python_type is uuid.UUID, table


def test_foreign_keys_are_declared(migrated_engine: Engine) -> None:
    inspector = inspect(migrated_engine)

    expected = {
        "turns": [("call_id", "calls")],
        "tool_calls": [("turn_id", "turns")],
        "appointments": [("call_id", "calls"), ("service_id", "services")],
    }
    for table, pairs in expected.items():
        actual = {
            (key["constrained_columns"][0], key["referred_table"])
            for key in inspector.get_foreign_keys(table)
        }
        assert set(pairs) <= actual, table


# --- what a call spent (milestone 8) --------------------------------------


def _cost(call: Call, **fields) -> CallCost:
    values = {
        "call_id": call.id,
        "component": CostComponent.LLM,
        "provider": "anthropic",
        "input_units": Decimal("100"),
        "unit_type": "tokens",
    }
    values.update(fields)
    return CallCost(**values)


def test_a_cost_row_defaults_to_no_price_and_no_cost(session: Session, call) -> None:
    """Unknown, and visibly so. Not free."""
    row = _cost(call)
    session.add(row)
    session.commit()

    assert row.cost_usd is None
    assert row.unit_price_usd is None


def test_a_cost_row_defaults_to_no_output_units_and_no_metadata(
    session: Session, call
) -> None:
    row = _cost(call)
    session.add(row)
    session.commit()
    session.refresh(row)

    assert row.output_units == Decimal("0.000")
    assert row.details == {}


def test_a_cost_row_keeps_six_decimal_places_of_money(
    session: Session, call
) -> None:
    row = _cost(call, cost_usd=Decimal("0.000001"))
    session.add(row)
    session.commit()
    session.refresh(row)

    assert row.cost_usd == Decimal("0.000001")


def test_a_unit_price_keeps_twelve_decimal_places(session: Session, call) -> None:
    """A per-token price is a millionth of a dollar; per-millisecond is less."""
    row = _cost(call, unit_price_usd=Decimal("0.000000000001"))
    session.add(row)
    session.commit()
    session.refresh(row)

    assert row.unit_price_usd == Decimal("0.000000000001")


def test_a_cost_row_may_belong_to_the_whole_call(session: Session, call) -> None:
    row = _cost(call, component=CostComponent.TELEPHONY, unit_type="duration_ms",
                provider="twilio")
    session.add(row)
    session.commit()

    assert row.turn_id is None


def test_a_cost_row_needs_a_call_that_exists(session: Session) -> None:
    session.add(
        CallCost(
            call_id=uuid.uuid4(),
            component=CostComponent.LLM,
            provider="anthropic",
            input_units=Decimal("1"),
            unit_type="tokens",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_the_component_enum_exists_in_the_database(migrated_engine: Engine) -> None:
    with migrated_engine.connect() as connection:
        values = connection.execute(
            text(
                "SELECT enumlabel FROM pg_enum JOIN pg_type ON pg_type.oid = "
                "enumtypid WHERE typname = 'cost_component' ORDER BY enumsortorder"
            )
        ).scalars().all()

    assert values == ["llm", "stt", "tts", "telephony"]


def test_the_cost_table_is_indexed_by_call_and_component(
    migrated_engine: Engine,
) -> None:
    names = {
        index["name"] for index in inspect(migrated_engine).get_indexes("call_costs")
    }

    assert "ix_call_costs_call_id" in names
    assert "ix_call_costs_call_id_component" in names


def test_the_two_uniqueness_rules_are_indexes_in_the_database(
    migrated_engine: Engine,
) -> None:
    """One per turn and component; one per call and component without a turn."""
    indexes = {
        index["name"]: index
        for index in inspect(migrated_engine).get_indexes("call_costs")
    }

    assert indexes["uq_call_costs_turn_component"]["unique"] is True
    assert indexes["uq_call_costs_call_component_without_turn"]["unique"] is True


def test_the_call_level_index_is_partial(migrated_engine: Engine) -> None:
    """Without the `WHERE`, it would forbid a second turn-level row too."""
    with migrated_engine.connect() as connection:
        definition = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes WHERE indexname = "
                "'uq_call_costs_call_component_without_turn'"
            )
        ).scalar_one()

    assert "WHERE (turn_id IS NULL)" in definition


def test_two_turn_level_rows_for_one_call_are_allowed(
    session: Session, call
) -> None:
    """Which is why the call-level index has to be partial."""
    first = Turn(call_id=call.id, role=TurnRole.AGENT, text="One")
    second = Turn(call_id=call.id, role=TurnRole.AGENT, text="Two")
    session.add_all([first, second])
    session.commit()

    session.add_all(
        [_cost(call, turn_id=first.id), _cost(call, turn_id=second.id)]
    )
    session.commit()

    assert len(session.execute(select(CallCost)).scalars().all()) == 2
