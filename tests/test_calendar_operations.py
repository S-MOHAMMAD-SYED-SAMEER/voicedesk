"""Booking, rescheduling and cancelling — against real PostgreSQL."""

import uuid
from datetime import UTC, datetime, time, timedelta

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.calendar import (
    AppointmentAlreadyCancelled,
    AppointmentNotFound,
    AppointmentNotReschedulable,
    CalendarService,
    InvalidInterval,
    InvalidServiceDuration,
    OutsideBusinessHours,
    ServiceNotFound,
    SlotUnavailable,
)
from app.config import Settings
from app.models import Appointment, AppointmentStatus, BusinessHours, Call, Service

MONDAY = datetime(2026, 3, 2, tzinfo=UTC).date()


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC)


def _book(calendar: CalendarService, service: Service, start: datetime, **kw):
    return calendar.book(
        service_id=service.id,
        starts_at=start,
        customer_name=kw.pop("customer_name", "Ada Lovelace"),
        phone=kw.pop("phone", "+447700900123"),
        **kw,
    )


# --- booking ---------------------------------------------------------------


def test_a_booking_is_persisted(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))

    stored = session.execute(select(Appointment)).scalar_one()
    assert stored.id == appointment.id
    assert stored.customer_name == "Ada Lovelace"
    assert stored.status is AppointmentStatus.BOOKED


def test_a_booking_occupies_the_whole_service_duration(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))

    assert appointment.starts_at == _at(10)
    assert appointment.ends_at == _at(10, 30)


def test_a_booking_copies_the_staff_member_from_the_service(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """The exclusion constraint needs it on the row it guards."""
    appointment = _book(calendar, haircut, _at(10))

    assert appointment.staff_id == haircut.staff_id == "sam"


def test_a_booking_can_record_the_call_that_made_it(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    call = Call(from_number="+447700900123", to_number="+441173450000")
    session.add(call)
    session.commit()

    appointment = _book(calendar, haircut, _at(10), call_id=call.id)

    assert appointment.call_id == call.id


def test_booking_removes_the_slot_from_availability(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    assert calendar.is_available(haircut.id, _at(10)) is True

    _book(calendar, haircut, _at(10))

    assert calendar.is_available(haircut.id, _at(10)) is False


def test_an_overlapping_booking_is_refused(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    _book(calendar, haircut, _at(10))

    with pytest.raises(SlotUnavailable, match="sam"):
        _book(calendar, haircut, _at(10, 15))


def test_an_adjacent_booking_is_allowed(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    _book(calendar, haircut, _at(10))

    _book(calendar, haircut, _at(10, 30))

    assert len(session.execute(select(Appointment)).scalars().all()) == 2


def test_two_staff_may_be_booked_at_once(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    other = Service(name="Massage", duration_minutes=30, staff_id="jo")
    session.add(other)
    session.commit()

    _book(calendar, haircut, _at(10))
    _book(calendar, other, _at(10))

    assert len(session.execute(select(Appointment)).scalars().all()) == 2


def test_the_session_survives_a_refused_booking(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """A savepoint rolls back only the refused write."""
    _book(calendar, haircut, _at(10))

    with pytest.raises(SlotUnavailable):
        _book(calendar, haircut, _at(10, 15))

    _book(calendar, haircut, _at(11))
    assert len(session.execute(select(Appointment)).scalars().all()) == 2


def test_booking_outside_business_hours_is_refused(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    with pytest.raises(OutsideBusinessHours):
        _book(calendar, haircut, _at(8))


def test_booking_that_would_overrun_closing_is_refused(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    with pytest.raises(OutsideBusinessHours):
        _book(calendar, haircut, _at(16, 45))


def test_booking_an_unknown_service_is_refused(
    calendar: CalendarService, open_weekdays
) -> None:
    with pytest.raises(ServiceNotFound):
        calendar.book(
            service_id=uuid.uuid4(),
            starts_at=_at(10),
            customer_name="Ada",
            phone="+1",
        )


def test_booking_an_inactive_service_is_refused(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    haircut.active = False
    session.commit()

    with pytest.raises(ServiceNotFound, match="not active"):
        _book(calendar, haircut, _at(10))


def test_a_naive_start_time_is_refused(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """A wall-clock time on its own does not name an instant."""
    with pytest.raises(InvalidInterval, match="timezone-aware"):
        _book(calendar, haircut, datetime(2026, 3, 2, 10))


def test_the_database_refuses_a_service_with_no_duration(session: Session) -> None:
    """The primary guard: such a service cannot be created at all."""
    from sqlalchemy.exc import IntegrityError

    session.add(Service(name="Odd", duration_minutes=0, staff_id="sam"))

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_the_calendar_also_guards_against_a_non_positive_duration(
    calendar: CalendarService,
) -> None:
    """Defence in depth.

    The check constraint above makes a zero-duration service unreachable
    through the database, so the guard is exercised against a transient
    service rather than a stored one — it exists so that a future change to
    the schema cannot silently produce zero-length appointments.
    """
    transient = Service(name="Odd", duration_minutes=0, staff_id="sam")

    with pytest.raises(InvalidServiceDuration):
        calendar._duration(transient)


# --- rescheduling ----------------------------------------------------------


def test_a_reschedule_moves_the_same_appointment(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """The audit meaning survives: same id, same creation time, same call."""
    appointment = _book(calendar, haircut, _at(10))
    original_id, created_at = appointment.id, appointment.created_at

    moved = calendar.reschedule(appointment.id, _at(14))

    assert moved.id == original_id
    assert moved.created_at == created_at
    assert moved.starts_at == _at(14)
    assert moved.ends_at == _at(14, 30)
    assert moved.status is AppointmentStatus.BOOKED


def test_a_reschedule_frees_the_old_slot(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))

    calendar.reschedule(appointment.id, _at(14))

    assert calendar.is_available(haircut.id, _at(10)) is True
    assert calendar.is_available(haircut.id, _at(14)) is False


def test_a_reschedule_onto_a_taken_slot_is_refused(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    first = _book(calendar, haircut, _at(10))
    _book(calendar, haircut, _at(14))

    with pytest.raises(SlotUnavailable):
        calendar.reschedule(first.id, _at(14, 15))


def test_a_reschedule_may_overlap_the_appointment_s_own_slot(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """Nudging 10:00 to 10:15 must not collide with itself."""
    appointment = _book(calendar, haircut, _at(10))

    moved = calendar.reschedule(appointment.id, _at(10, 15))

    assert moved.starts_at == _at(10, 15)


def test_a_reschedule_outside_hours_is_refused(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))

    with pytest.raises(OutsideBusinessHours):
        calendar.reschedule(appointment.id, _at(19))


def test_a_cancelled_appointment_cannot_be_rescheduled(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))
    calendar.cancel(appointment.id)

    with pytest.raises(AppointmentNotReschedulable, match="cancelled"):
        calendar.reschedule(appointment.id, _at(14))


def test_rescheduling_an_unknown_appointment_is_refused(
    calendar: CalendarService, open_weekdays
) -> None:
    with pytest.raises(AppointmentNotFound):
        calendar.reschedule(uuid.uuid4(), _at(10))


# --- cancelling ------------------------------------------------------------


def test_cancelling_marks_the_appointment_rather_than_deleting_it(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))

    cancelled = calendar.cancel(appointment.id)

    assert cancelled.status is AppointmentStatus.CANCELLED
    assert session.get(Appointment, appointment.id) is not None


def test_cancelling_gives_the_slot_back(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))
    assert calendar.is_available(haircut.id, _at(10)) is False

    calendar.cancel(appointment.id)

    assert calendar.is_available(haircut.id, _at(10)) is True


def test_the_slot_can_be_booked_again_after_cancelling(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """The exclusion constraint is partial, so the cancelled row does not block."""
    appointment = _book(calendar, haircut, _at(10))
    calendar.cancel(appointment.id)

    replacement = _book(calendar, haircut, _at(10), customer_name="Bob")

    assert replacement.id != appointment.id


def test_cancelling_twice_is_refused(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    appointment = _book(calendar, haircut, _at(10))
    calendar.cancel(appointment.id)

    with pytest.raises(AppointmentAlreadyCancelled):
        calendar.cancel(appointment.id)


def test_cancelling_an_unknown_appointment_is_refused(
    calendar: CalendarService, open_weekdays
) -> None:
    with pytest.raises(AppointmentNotFound):
        calendar.cancel(uuid.uuid4())


# --- the race the constraint exists for ------------------------------------


def test_two_callers_racing_for_one_slot_and_only_one_wins(
    migrated_engine: Engine,
    session: Session,
    calendar_settings: Settings,
    open_weekdays,
    haircut: Service,
) -> None:
    """Both check, both find it free, both book. One must lose.

    This is the case an availability check cannot cover: neither transaction
    can see the other's uncommitted row, so only the database can arbitrate.
    """
    with Session(migrated_engine) as first, Session(migrated_engine) as second:
        alice = CalendarService(first, calendar_settings)
        bob = CalendarService(second, calendar_settings)

        # Both look before either writes, and both are told yes.
        assert alice.is_available(haircut.id, _at(10)) is True
        assert bob.is_available(haircut.id, _at(10)) is True

        alice.book(
            service_id=haircut.id,
            starts_at=_at(10),
            customer_name="Alice",
            phone="+1",
        )

        with pytest.raises(SlotUnavailable):
            bob.book(
                service_id=haircut.id,
                starts_at=_at(10, 15),
                customer_name="Bob",
                phone="+2",
            )

    booked = session.execute(
        select(Appointment).filter_by(status=AppointmentStatus.BOOKED)
    ).scalars().all()
    assert len(booked) == 1
    assert booked[0].customer_name == "Alice"


def test_the_exclusion_constraint_is_still_the_one_from_milestone_one(
    migrated_engine: Engine,
) -> None:
    """M2 must not have weakened or replaced it."""
    from sqlalchemy import text

    with migrated_engine.connect() as connection:
        definition = connection.execute(
            text(
                "select pg_get_constraintdef(oid) from pg_constraint "
                "where conname = 'ck_appointments_no_double_booking'"
            )
        ).scalar_one()

    assert "EXCLUDE USING gist" in definition
    assert "staff_id WITH =" in definition
    assert "'[)'" in definition
    assert "status = 'booked'" in definition
