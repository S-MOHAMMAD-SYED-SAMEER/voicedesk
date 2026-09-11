"""Which start times are actually free.

Deterministic and database-backed: candidate times come from the configured
opening periods and slot granularity, and are then removed by the appointments
already on that staff member's diary. Nothing here guesses, and nothing here
asks a model — the future dialogue layer must take availability from this
code, never invent it.

The result is advisory. It is true when it is computed and can be stale a
moment later, which is why booking is guarded by a database constraint rather
than by this answer. See `app/calendar/service.py`.
"""

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.calendar.hours import OpeningPeriod, periods_for
from app.models import Appointment, AppointmentStatus


@dataclass(frozen=True)
class Slot:
    """A bookable interval, half-open: `[starts_at, ends_at)`."""

    starts_at: datetime
    ends_at: datetime

    def overlaps(self, other_start: datetime, other_end: datetime) -> bool:
        """Half-open overlap: touching at an endpoint is not overlapping."""
        return self.starts_at < other_end and other_start < self.ends_at


def candidate_slots(
    period: OpeningPeriod, duration: timedelta, granularity: timedelta
) -> list[Slot]:
    """Every start time on the grid that leaves room for the full duration.

    The grid is measured from the period's own opening time, not from
    midnight: a shop opening at 09:10 offers 09:10, 09:25, … rather than
    times it is shut for.
    """
    slots: list[Slot] = []
    cursor = period.opens_at
    while cursor + duration <= period.closes_at:
        slots.append(Slot(cursor, cursor + duration))
        cursor += granularity
    return slots


def booked_intervals(
    session: Session, staff_id: str, window_start: datetime, window_end: datetime
) -> list[tuple[datetime, datetime]]:
    """Booked appointments for one staff member touching the window.

    Cancelled appointments are excluded, which is what lets a cancellation
    give a slot back.
    """
    rows = (
        session.execute(
            select(Appointment.starts_at, Appointment.ends_at)
            .where(Appointment.staff_id == staff_id)
            .where(Appointment.status == AppointmentStatus.BOOKED)
            .where(Appointment.starts_at < window_end)
            .where(Appointment.ends_at > window_start)
            .order_by(Appointment.starts_at)
        )
        .tuples()
        .all()
    )
    return [(starts_at, ends_at) for starts_at, ends_at in rows]


def free_slots(
    session: Session,
    *,
    staff_id: str,
    day: date,
    duration: timedelta,
    granularity: timedelta,
    timezone: ZoneInfo,
    exclude_appointment_id: uuid.UUID | None = None,
) -> list[Slot]:
    """Bookable start times for one staff member on one local day.

    `exclude_appointment_id` lets a reschedule see the slots its own current
    booking is occupying, which it is entitled to keep.
    """
    periods = periods_for(session, day, timezone)
    if not periods:
        return []

    candidates = [
        slot
        for period in periods
        for slot in candidate_slots(period, duration, granularity)
    ]
    if not candidates:
        return []

    taken = _taken_intervals(
        session,
        staff_id=staff_id,
        window_start=periods[0].opens_at,
        window_end=periods[-1].closes_at,
        exclude_appointment_id=exclude_appointment_id,
    )

    return [
        slot
        for slot in candidates
        if not any(slot.overlaps(start, end) for start, end in taken)
    ]


def _taken_intervals(
    session: Session,
    *,
    staff_id: str,
    window_start: datetime,
    window_end: datetime,
    exclude_appointment_id: uuid.UUID | None,
) -> list[tuple[datetime, datetime]]:
    if exclude_appointment_id is None:
        return booked_intervals(session, staff_id, window_start, window_end)

    rows = (
        session.execute(
            select(Appointment.starts_at, Appointment.ends_at)
            .where(Appointment.staff_id == staff_id)
            .where(Appointment.status == AppointmentStatus.BOOKED)
            .where(Appointment.starts_at < window_end)
            .where(Appointment.ends_at > window_start)
            .where(Appointment.id != exclude_appointment_id)
            .order_by(Appointment.starts_at)
        )
        .tuples()
        .all()
    )
    return [(starts_at, ends_at) for starts_at, ends_at in rows]


def is_free(
    session: Session,
    *,
    staff_id: str,
    starts_at: datetime,
    ends_at: datetime,
    exclude_appointment_id: uuid.UUID | None = None,
) -> bool:
    """Is this exact interval clear of that staff member's bookings?

    Advisory only — see the module docstring.
    """
    taken = _taken_intervals(
        session,
        staff_id=staff_id,
        window_start=starts_at,
        window_end=ends_at,
        exclude_appointment_id=exclude_appointment_id,
    )
    return not taken
