"""`cancel` — giving a slot back without losing the record."""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.models import Appointment, AppointmentStatus, Service
from app.tools import ToolContext, book_appointment, cancel, check_availability


def _at(hour: int) -> str:
    return datetime(2026, 3, 2, hour, tzinfo=UTC).isoformat()


def _booked(context: ToolContext, hour: int = 10) -> str:
    result = book_appointment(
        context,
        service_name="Haircut",
        starts_at=_at(hour),
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )
    assert result.success, result.error
    return result.data["appointment_id"]


def test_cancelling_reports_the_new_status(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)

    result = cancel(tools, appointment_id=appointment_id)

    assert result.success, result.error
    assert result.data["appointment_id"] == appointment_id
    assert result.data["status"] == "cancelled"


def test_the_row_is_kept_not_deleted(
    session: Session, tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)
    cancel(tools, appointment_id=appointment_id)

    stored = session.get(Appointment, uuid.UUID(appointment_id))
    assert stored is not None
    assert stored.status is AppointmentStatus.CANCELLED


def test_the_slot_becomes_available_again(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)
    cancel(tools, appointment_id=appointment_id)

    starts = [
        slot["starts_at"]
        for slot in check_availability(
            tools, service_name="Haircut", day="2026-03-02"
        ).data["slots"]
    ]
    assert "2026-03-02T10:00:00+00:00" in starts


def test_cancelling_twice_is_refused(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)
    cancel(tools, appointment_id=appointment_id)

    result = cancel(tools, appointment_id=appointment_id)

    assert not result.success
    assert "already cancelled" in result.error


def test_an_unknown_appointment_fails(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = cancel(tools, appointment_id=str(uuid.uuid4()))

    assert not result.success
    assert "No appointment" in result.error


def test_an_identifier_that_is_not_one_fails_clearly(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = cancel(tools, appointment_id="that appointment")

    assert not result.success
    assert "must be an identifier" in result.error


def test_a_uuid_object_is_accepted_as_well_as_a_string(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)

    result = cancel(tools, appointment_id=uuid.UUID(appointment_id))

    assert result.success, result.error
