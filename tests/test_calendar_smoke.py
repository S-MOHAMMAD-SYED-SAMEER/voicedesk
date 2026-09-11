"""One deterministic pass through the whole calendar core.

service → business hours → availability → book → availability changes →
cancel → availability returns.
"""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.calendar import CalendarService
from app.models import Appointment, AppointmentStatus, Service

MONDAY = datetime(2026, 3, 2, tzinfo=UTC).date()
TEN = datetime(2026, 3, 2, 10, tzinfo=UTC)


def test_the_whole_calendar_core_end_to_end(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    # A service exists, performed by one staff member, inside business hours.
    assert haircut.duration_minutes == 30
    assert haircut.staff_id == "sam"

    # Availability is offered on the configured grid.
    before = [slot.starts_at for slot in calendar.available_slots(haircut.id, MONDAY)]
    assert TEN in before
    assert len(before) == 31

    # Booking takes the slot, and the ones that would overlap it.
    appointment = calendar.book(
        service_id=haircut.id,
        starts_at=TEN,
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )
    after_booking = [
        slot.starts_at for slot in calendar.available_slots(haircut.id, MONDAY)
    ]
    assert TEN not in after_booking
    assert len(after_booking) == len(before) - 3  # 09:45, 10:00, 10:15

    # Cancelling gives every one of them back.
    calendar.cancel(appointment.id)
    after_cancelling = [
        slot.starts_at for slot in calendar.available_slots(haircut.id, MONDAY)
    ]
    assert after_cancelling == before

    # The appointment is kept, not deleted: the record of what happened stands.
    stored = session.get(Appointment, appointment.id)
    assert stored is not None
    assert stored.status is AppointmentStatus.CANCELLED
