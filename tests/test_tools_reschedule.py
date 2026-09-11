"""`reschedule` — the same commitment, at a new time."""

import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.models import AppointmentStatus, Service
from app.tools import ToolContext, book_appointment, cancel, reschedule


def _at(hour: int, minute: int = 0) -> str:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC).isoformat()


def _booked(context: ToolContext, hour: int = 10, **overrides) -> str:
    arguments = {
        "service_name": "Haircut",
        "starts_at": _at(hour),
        "customer_name": "Ada Lovelace",
        "phone": "+447700900123",
    }
    arguments.update(overrides)
    result = book_appointment(context, **arguments)
    assert result.success, result.error
    return result.data["appointment_id"]


def test_an_appointment_moves_and_keeps_its_identity(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)

    result = reschedule(tools, appointment_id=appointment_id, new_starts_at=_at(14))

    assert result.success, result.error
    assert result.data["appointment_id"] == appointment_id
    assert result.data["starts_at"] == "2026-03-02T14:00:00+00:00"
    assert result.data["ends_at"] == "2026-03-02T14:30:00+00:00"


def test_the_old_time_becomes_available_again(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    from app.tools import check_availability

    appointment_id = _booked(tools)
    reschedule(tools, appointment_id=appointment_id, new_starts_at=_at(14))

    starts = [
        slot["starts_at"]
        for slot in check_availability(
            tools, service_name="Haircut", day="2026-03-02"
        ).data["slots"]
    ]
    assert "2026-03-02T10:00:00+00:00" in starts
    assert "2026-03-02T14:00:00+00:00" not in starts


def test_moving_onto_a_taken_slot_is_a_failed_result(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools, hour=10)
    _booked(tools, hour=14, customer_name="Grace Hopper")

    result = reschedule(tools, appointment_id=appointment_id, new_starts_at=_at(14))

    assert not result.success
    assert result.data["slot_taken"] is True


def test_a_move_outside_business_hours_is_refused(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)

    result = reschedule(tools, appointment_id=appointment_id, new_starts_at=_at(21))

    assert not result.success
    assert "opening period" in result.error


def test_a_cancelled_appointment_cannot_be_moved(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)
    cancel(tools, appointment_id=appointment_id)

    result = reschedule(tools, appointment_id=appointment_id, new_starts_at=_at(14))

    assert not result.success
    assert "cannot be moved" in result.error


def test_an_unknown_appointment_fails(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = reschedule(
        tools, appointment_id=str(uuid.uuid4()), new_starts_at=_at(14)
    )

    assert not result.success
    assert "No appointment" in result.error


def test_an_identifier_that_is_not_one_fails_clearly(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = reschedule(tools, appointment_id="the haircut one", new_starts_at=_at(14))

    assert not result.success
    assert "must be an identifier" in result.error


def test_an_unparsable_new_time_names_its_own_field(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)

    result = reschedule(
        tools, appointment_id=appointment_id, new_starts_at="a bit later"
    )

    assert not result.success
    assert "new_starts_at" in result.error


def test_a_naive_new_time_is_read_in_the_business_timezone(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    appointment_id = _booked(tools)

    result = reschedule(
        tools, appointment_id=appointment_id, new_starts_at="2026-03-02T14:00:00"
    )

    assert result.success, result.error
    assert result.data["starts_at"] == "2026-03-02T14:00:00+00:00"


def test_a_moved_appointment_is_still_booked(
    session: Session, tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    from app.models import Appointment

    appointment_id = _booked(tools)
    reschedule(tools, appointment_id=appointment_id, new_starts_at=_at(14))

    stored = session.get(Appointment, uuid.UUID(appointment_id))
    assert stored.status is AppointmentStatus.BOOKED
