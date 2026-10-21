"""Finding an existing booking by name and number, and the rule that only found or just-booked appointments can be changed."""

from datetime import UTC, datetime

import pytest

from app.calendar.lookup import phone_key
from app.clock import fixed_clock
from app.dialogue import UNKNOWN_APPOINTMENT, Conversation
from app.models import Appointment
from app.tools import ToolContext, find_appointments
from tests.conftest import FakeModel, say, use_tools

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
TEN = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
TWO = datetime(2026, 3, 2, 14, 0, tzinfo=UTC)


def _context(session, settings, now=NOW) -> ToolContext:
    return ToolContext(session=session, settings=settings, clock=fixed_clock(now))


def _book(session, settings, service, *, name="Ada Lovelace", phone="+447700900123", at=TEN):
    from app.calendar import CalendarService

    return CalendarService(session, settings, clock=fixed_clock(NOW)).book(
        service_id=service.id, starts_at=at, customer_name=name, phone=phone
    )


# --- the lookup ------------------------------------------------------------


def test_a_booking_is_found_by_name_and_number(session, calendar_settings, open_weekdays, haircut) -> None:
    booked = _book(session, calendar_settings, haircut)

    result = find_appointments(
        _context(session, calendar_settings),
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )

    assert result.success and result.data["found"] == 1
    item = result.data["appointments"][0]
    assert item["appointment_id"] == str(booked.id)
    assert item["service"] == "Haircut"
    assert item["starts_at"] == "2026-03-02T10:00:00+00:00"


@pytest.mark.parametrize(
    "spoken",
    ["07700 900123", "+44 7700 900123", "+44 (0)7700-900-123", "447700900123", "7700900123"],
)
def test_a_number_matches_however_it_is_written(session, calendar_settings, open_weekdays, haircut, spoken) -> None:
    _book(session, calendar_settings, haircut)

    result = find_appointments(
        _context(session, calendar_settings), customer_name="Ada Lovelace", phone=spoken
    )

    assert result.data["found"] == 1, spoken


def test_a_name_matches_without_regard_to_case(session, calendar_settings, open_weekdays, haircut) -> None:
    _book(session, calendar_settings, haircut)

    result = find_appointments(
        _context(session, calendar_settings), customer_name="  ada LOVELACE ", phone="07700900123"
    )

    assert result.data["found"] == 1


def test_the_right_number_with_the_wrong_name_finds_nothing(session, calendar_settings, open_weekdays, haircut) -> None:
    _book(session, calendar_settings, haircut)

    result = find_appointments(
        _context(session, calendar_settings), customer_name="Bea Bramble", phone="+447700900123"
    )

    assert result.success and result.data["appointments"] == []


def test_the_right_name_with_the_wrong_number_finds_nothing(session, calendar_settings, open_weekdays, haircut) -> None:
    _book(session, calendar_settings, haircut)

    result = find_appointments(
        _context(session, calendar_settings), customer_name="Ada Lovelace", phone="+447700900999"
    )

    assert result.data["appointments"] == []


def test_a_cancelled_booking_is_not_found(session, calendar_settings, open_weekdays, haircut) -> None:
    from app.calendar import CalendarService

    booked = _book(session, calendar_settings, haircut)
    CalendarService(session, calendar_settings, clock=fixed_clock(NOW)).cancel(booked.id)

    result = find_appointments(
        _context(session, calendar_settings), customer_name="Ada Lovelace", phone="+447700900123"
    )

    assert result.data["appointments"] == []


def test_a_booking_that_has_already_happened_is_not_found(session, calendar_settings, open_weekdays, haircut) -> None:
    _book(session, calendar_settings, haircut)

    result = find_appointments(
        _context(session, calendar_settings, now=datetime(2026, 3, 2, 11, 0, tzinfo=UTC)),
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )

    assert result.data["appointments"] == []


def test_several_bookings_come_back_earliest_first(session, calendar_settings, open_weekdays, haircut) -> None:
    _book(session, calendar_settings, haircut, at=TWO)
    _book(session, calendar_settings, haircut, at=TEN)

    result = find_appointments(
        _context(session, calendar_settings), customer_name="Ada Lovelace", phone="+447700900123"
    )

    assert [item["starts_at"][11:16] for item in result.data["appointments"]] == ["10:00", "14:00"]


def test_one_callers_booking_is_not_shown_to_another(session, calendar_settings, open_weekdays, haircut) -> None:
    _book(session, calendar_settings, haircut, name="Ada Lovelace", phone="+447700900123", at=TEN)
    _book(session, calendar_settings, haircut, name="Bea Bramble", phone="+447700900456", at=TWO)

    result = find_appointments(
        _context(session, calendar_settings), customer_name="Bea Bramble", phone="+447700900456"
    )

    assert [item["starts_at"][11:16] for item in result.data["appointments"]] == ["14:00"]


def test_a_number_with_too_few_digits_is_refused(session, calendar_settings) -> None:
    result = find_appointments(
        _context(session, calendar_settings), customer_name="Ada Lovelace", phone="123"
    )

    assert not result.success and "digits" in result.error


def test_a_missing_name_is_refused(session, calendar_settings) -> None:
    result = find_appointments(
        _context(session, calendar_settings), customer_name=" ", phone="+447700900123"
    )

    assert not result.success


def test_phone_key_is_the_last_nine_digits() -> None:
    assert phone_key("+44 (0)7700 900123") == "700900123"
    assert phone_key("12345") == "12345"


# --- the identity rule: only an identifier this conversation was given ------


def _conversation(session, call, model, settings) -> Conversation:
    return Conversation(session, call, model, settings, clock=fixed_clock(NOW))


def test_an_identifier_nobody_gave_is_refused_for_cancel(
    session, call, calendar_settings, open_weekdays, haircut
) -> None:
    """The booking exists; the model never found it. Naming its id is not enough."""
    booked = _book(session, calendar_settings, haircut)
    model = FakeModel(use_tools(("cancel", {"appointment_id": str(booked.id)})), say("Sorry."))

    result = _conversation(session, call, model, calendar_settings).send("Cancel it.")

    record = result.tool_calls[0]
    assert not record.success and record.error == UNKNOWN_APPOINTMENT
    session.expire_all()
    assert session.get(Appointment, booked.id).status.value == "booked"


def test_an_identifier_nobody_gave_is_refused_for_reschedule(
    session, call, calendar_settings, open_weekdays, haircut
) -> None:
    booked = _book(session, calendar_settings, haircut)
    model = FakeModel(
        use_tools(
            ("reschedule", {"appointment_id": str(booked.id), "new_starts_at": TWO.isoformat()})
        ),
        say("Sorry."),
    )

    result = _conversation(session, call, model, calendar_settings).send("Move it.")

    assert result.tool_calls[0].error == UNKNOWN_APPOINTMENT
    session.expire_all()
    assert session.get(Appointment, booked.id).starts_at == TEN


def test_after_a_matching_lookup_the_booking_can_be_moved(
    session, call, calendar_settings, open_weekdays, haircut
) -> None:
    booked = _book(session, calendar_settings, haircut)
    model = FakeModel(
        use_tools(("find_appointments", {"customer_name": "Ada Lovelace", "phone": "07700 900123"})),
        use_tools(("reschedule", {"appointment_id": str(booked.id), "new_starts_at": TWO.isoformat()})),
        say("That is moved."),
    )

    result = _conversation(session, call, model, calendar_settings).send("Move my haircut to two.")

    assert [r.tool_name for r in result.tool_calls] == ["find_appointments", "reschedule"]
    assert all(r.success for r in result.tool_calls)
    session.expire_all()
    assert session.get(Appointment, booked.id).starts_at == TWO


def test_after_a_matching_lookup_the_booking_can_be_cancelled(
    session, call, calendar_settings, open_weekdays, haircut
) -> None:
    booked = _book(session, calendar_settings, haircut)
    model = FakeModel(
        use_tools(("find_appointments", {"customer_name": "Ada Lovelace", "phone": "+447700900123"})),
        use_tools(("cancel", {"appointment_id": str(booked.id)})),
        say("That is cancelled."),
    )

    result = _conversation(session, call, model, calendar_settings).send("Cancel my haircut.")

    assert all(r.success for r in result.tool_calls)
    session.expire_all()
    assert session.get(Appointment, booked.id).status.value == "cancelled"


def test_a_lookup_that_matched_nothing_unlocks_nothing(
    session, call, calendar_settings, open_weekdays, haircut
) -> None:
    booked = _book(session, calendar_settings, haircut)
    model = FakeModel(
        use_tools(("find_appointments", {"customer_name": "Bea Bramble", "phone": "+447700900456"})),
        use_tools(("cancel", {"appointment_id": str(booked.id)})),
        say("Sorry."),
    )

    result = _conversation(session, call, model, calendar_settings).send("Cancel Ada's haircut.")

    assert result.tool_calls[0].data["appointments"] == []
    assert result.tool_calls[1].error == UNKNOWN_APPOINTMENT
    session.expire_all()
    assert session.get(Appointment, booked.id).status.value == "booked"


def test_a_booking_made_in_this_call_can_be_changed_without_a_lookup(
    session, call, calendar_settings, open_weekdays, haircut
) -> None:
    model = FakeModel(
        use_tools(("check_availability", {"service_name": "Haircut", "day": "2026-03-02"})),
        use_tools(
            (
                "book_appointment",
                {
                    "service_name": "Haircut",
                    "starts_at": TEN.isoformat(),
                    "customer_name": "Ada Lovelace",
                    "phone": "+447700900123",
                },
            )
        ),
        say("Booked."),
    )
    conversation = _conversation(session, call, model, calendar_settings)
    first = conversation.send("Book me a haircut on Monday at ten.")
    booked_id = first.booked_appointment_id
    conversation_model = FakeModel(use_tools(("cancel", {"appointment_id": booked_id})), say("Cancelled."))
    conversation._model = conversation_model  # the same conversation, a new script

    second = conversation.send("Actually, cancel that.")

    assert second.tool_calls[0].success
