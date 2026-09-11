"""Available slots: what the future dialogue layer will be told is free."""

from datetime import UTC, datetime, time, timedelta

import pytest
from sqlalchemy.orm import Session

from app.calendar import CalendarService, ServiceNotFound
from app.config import Settings
from app.models import Appointment, AppointmentStatus, BusinessHours, Service

MONDAY = datetime(2026, 3, 2, tzinfo=UTC).date()
SATURDAY = datetime(2026, 3, 7, tzinfo=UTC).date()


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC)


def _book(session: Session, service: Service, start: datetime, minutes: int = 30,
          status: AppointmentStatus = AppointmentStatus.BOOKED) -> Appointment:
    """Insert directly, so availability is tested against data, not booking."""
    appointment = Appointment(
        customer_name="Ada",
        phone="+447700900123",
        service_id=service.id,
        staff_id=service.staff_id,
        starts_at=start,
        ends_at=start + timedelta(minutes=minutes),
        status=status,
    )
    session.add(appointment)
    session.commit()
    return appointment


def _starts(calendar: CalendarService, service: Service, day=MONDAY) -> list[datetime]:
    return [slot.starts_at for slot in calendar.available_slots(service.id, day)]


# --- an empty calendar -----------------------------------------------------


def test_an_empty_day_offers_the_whole_grid(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    starts = _starts(calendar, haircut)

    # 09:00 to 16:30 inclusive, every 15 minutes.
    assert starts[0] == _at(9)
    assert starts[-1] == _at(16, 30)
    assert len(starts) == 31
    assert starts[1] - starts[0] == timedelta(minutes=15)


def test_every_slot_leaves_room_for_the_whole_service(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """A 30-minute service is an interval, never a point in time."""
    slots = calendar.available_slots(haircut.id, MONDAY)

    assert all(slot.ends_at - slot.starts_at == timedelta(minutes=30) for slot in slots)
    assert all(slot.ends_at <= _at(17) for slot in slots)


def test_a_closed_day_offers_nothing(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    assert calendar.available_slots(haircut.id, SATURDAY) == []


def test_a_longer_service_offers_fewer_slots(
    session: Session, calendar: CalendarService, open_weekdays
) -> None:
    long_service = Service(name="Colour", duration_minutes=120, staff_id="sam")
    session.add(long_service)
    session.commit()

    starts = _starts(calendar, long_service)

    assert starts[0] == _at(9)
    assert starts[-1] == _at(15)  # 15:00-17:00 is the last that fits


def test_the_grid_is_measured_from_opening_not_midnight(
    session: Session, calendar: CalendarService, haircut: Service
) -> None:
    session.add(BusinessHours(weekday=0, opens_at=time(9, 10), closes_at=time(11)))
    session.commit()

    starts = _starts(calendar, haircut)

    assert starts[:3] == [_at(9, 10), _at(9, 25), _at(9, 40)]


def test_the_granularity_is_configurable(
    session: Session, open_weekdays, haircut: Service
) -> None:
    half_hourly = CalendarService(
        session,
        Settings(_env_file=None, business_timezone="UTC", slot_granularity_minutes=30),
    )

    starts = [slot.starts_at for slot in half_hourly.available_slots(haircut.id, MONDAY)]

    assert starts[:3] == [_at(9), _at(9, 30), _at(10)]
    assert len(starts) == 16


# --- appointments removing slots -------------------------------------------


def test_a_booked_appointment_removes_the_overlapping_slots(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    _book(session, haircut, _at(10))

    starts = _starts(calendar, haircut)

    # A 30-minute booking at 10:00 blocks every start from 09:45 to 10:15.
    assert _at(9, 30) in starts
    assert _at(9, 45) not in starts
    assert _at(10) not in starts
    assert _at(10, 15) not in starts
    assert _at(10, 30) in starts


def test_an_adjacent_appointment_leaves_the_touching_slot(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """The interval is half-open, so 10:30 is free when 10:00-10:30 is taken."""
    _book(session, haircut, _at(10))

    assert _at(10, 30) in _starts(calendar, haircut)


def test_a_cancelled_appointment_blocks_nothing(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    _book(session, haircut, _at(10), status=AppointmentStatus.CANCELLED)

    assert _at(10) in _starts(calendar, haircut)


def test_a_full_day_offers_nothing(
    session: Session, calendar: CalendarService, haircut: Service
) -> None:
    session.add(BusinessHours(weekday=0, opens_at=time(9), closes_at=time(10)))
    session.commit()
    _book(session, haircut, _at(9), minutes=60)

    assert calendar.available_slots(haircut.id, MONDAY) == []


def test_a_day_too_short_for_the_service_offers_nothing(
    session: Session, calendar: CalendarService, haircut: Service
) -> None:
    """Twenty open minutes cannot hold a thirty-minute service."""
    session.add(BusinessHours(weekday=0, opens_at=time(9), closes_at=time(9, 20)))
    session.commit()

    assert calendar.available_slots(haircut.id, MONDAY) == []


# --- staff scoping ---------------------------------------------------------


def test_another_staff_members_booking_is_irrelevant(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    other = Service(name="Massage", duration_minutes=60, staff_id="jo")
    session.add(other)
    session.commit()
    _book(session, other, _at(10), minutes=60)

    assert _at(10) in _starts(calendar, haircut)


def test_a_staff_members_other_service_does_block(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    """Availability follows the person, not the service."""
    same_staff = Service(name="Beard trim", duration_minutes=15, staff_id="sam")
    session.add(same_staff)
    session.commit()
    _book(session, same_staff, _at(10), minutes=15)

    assert _at(10) not in _starts(calendar, haircut)


# --- is_available ----------------------------------------------------------


def test_is_available_agrees_with_the_slot_list(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    _book(session, haircut, _at(10))

    assert calendar.is_available(haircut.id, _at(9, 30)) is True
    assert calendar.is_available(haircut.id, _at(10)) is False


def test_is_available_is_false_outside_hours(
    calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    assert calendar.is_available(haircut.id, _at(8)) is False
    assert calendar.is_available(haircut.id, _at(16, 45)) is False


def test_an_unknown_service_is_rejected(calendar: CalendarService) -> None:
    import uuid

    with pytest.raises(ServiceNotFound):
        calendar.available_slots(uuid.uuid4(), MONDAY)


def test_an_inactive_service_is_not_bookable(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    haircut.active = False
    session.commit()

    with pytest.raises(ServiceNotFound, match="not active"):
        calendar.available_slots(haircut.id, MONDAY)
