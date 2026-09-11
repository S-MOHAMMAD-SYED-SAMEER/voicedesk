"""`book_appointment` — commitments, and the ways they fail."""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Appointment, AppointmentStatus, Call, Service
from app.tools import ToolContext, book_appointment


def _at(hour: int, minute: int = 0) -> str:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC).isoformat()


def _book(context: ToolContext, **overrides):
    arguments = {
        "service_name": "Haircut",
        "starts_at": _at(10),
        "customer_name": "Ada Lovelace",
        "phone": "+447700900123",
    }
    arguments.update(overrides)
    return book_appointment(context, **arguments)


def test_a_booking_is_persisted(
    session: Session, tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools)

    assert result.success, result.error
    stored = session.execute(select(Appointment)).scalar_one()
    assert str(stored.id) == result.data["appointment_id"]
    assert stored.customer_name == "Ada Lovelace"
    assert stored.status is AppointmentStatus.BOOKED


def test_the_result_reports_the_whole_interval(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools)

    assert result.data["starts_at"] == "2026-03-02T10:00:00+00:00"
    assert result.data["ends_at"] == "2026-03-02T10:30:00+00:00"
    assert result.data["status"] == "booked"


def test_the_booking_is_attributed_to_the_call_in_context(
    session: Session, calendar_settings, open_weekdays, haircut: Service
) -> None:
    """The call comes from the context, so it cannot be chosen by an argument."""
    call = Call(
        direction="inbound",
        from_number="+447700900999",
        to_number="+441234567890",
    )
    session.add(call)
    session.commit()
    call_id = call.id

    context = ToolContext(
        session=session, settings=calendar_settings, call_id=call_id
    )
    result = _book(context)

    stored = session.execute(select(Appointment)).scalar_one()
    assert result.success, result.error
    assert stored.call_id == call_id


def test_a_taken_slot_is_a_failed_result_not_an_exception(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    assert _book(tools).success

    result = _book(tools, customer_name="Grace Hopper")

    assert not result.success
    assert result.data["slot_taken"] is True
    assert "sam" in result.error


def test_a_slot_freed_by_a_cancellation_can_be_booked_again(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    from app.tools import cancel

    first = _book(tools)
    cancel(tools, appointment_id=first.data["appointment_id"])

    second = _book(tools, customer_name="Grace Hopper")

    assert second.success, second.error


def test_a_time_outside_business_hours_is_refused(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools, starts_at=_at(20))

    assert not result.success
    assert "opening period" in result.error


def test_a_naive_time_is_read_in_the_business_timezone(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    """"Ten o'clock" from a caller means ten where the business is."""
    result = _book(tools, starts_at="2026-03-02T10:00:00")

    assert result.success, result.error
    assert result.data["starts_at"] == "2026-03-02T10:00:00+00:00"


def test_an_unparsable_time_fails_clearly(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools, starts_at="tomorrow morning")

    assert not result.success
    assert "ISO timestamp" in result.error


def test_a_missing_customer_name_is_refused(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools, customer_name="   ")

    assert not result.success
    assert "customer_name is required" in result.error


def test_a_missing_phone_is_refused(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools, phone="")

    assert not result.success
    assert "phone is required" in result.error


def test_an_unknown_service_books_nothing(
    session: Session, tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools, service_name="Hovercraft")

    assert not result.success
    assert result.data["services_offered"] == ["Haircut"]
    assert session.execute(select(Appointment)).first() is None


def test_an_inactive_service_cannot_be_booked(
    session: Session, tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    haircut.active = False
    session.commit()

    result = _book(tools)

    assert not result.success
    assert session.execute(select(Appointment)).first() is None


def test_a_service_name_is_matched_ignoring_case_and_padding(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = _book(tools, service_name="  haircut ")

    assert result.success, result.error
    assert result.data["service"] == "Haircut"


def test_a_duplicated_service_name_is_ambiguous_and_never_guessed(
    session: Session, tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    session.add(Service(name="haircut", duration_minutes=45, staff_id="alex"))
    session.commit()

    result = _book(tools)

    assert not result.success
    assert "ambiguous" in result.error
    assert session.execute(select(Appointment)).first() is None
