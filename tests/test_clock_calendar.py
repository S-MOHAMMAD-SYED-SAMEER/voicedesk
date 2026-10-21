"""The calendar knows what time it is, and refuses the past."""

from datetime import UTC, datetime, timedelta

import pytest

from app.calendar import CalendarService, SlotInThePast
from app.clock import fixed_clock, system_clock
from tests.conftest import SUITE_NOW

SUNDAY_NOON = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
MONDAY = "2026-03-02"


def _service(session, calendar_settings, now):
    return CalendarService(session, calendar_settings, clock=fixed_clock(now))


def test_a_slot_in_the_past_cannot_be_booked(session, calendar_settings, open_weekdays, haircut) -> None:
    calendar = _service(session, calendar_settings, datetime(2026, 3, 2, 12, 0, tzinfo=UTC))

    with pytest.raises(SlotInThePast):
        calendar.book(
            service_id=haircut.id,
            starts_at=datetime(2026, 3, 2, 10, 0, tzinfo=UTC),
            customer_name="Ada Lovelace",
            phone="+447700900123",
        )


def test_the_instant_it_is_now_is_also_past(session, calendar_settings, open_weekdays, haircut) -> None:
    now = datetime(2026, 3, 2, 10, 0, tzinfo=UTC)
    calendar = _service(session, calendar_settings, now)

    with pytest.raises(SlotInThePast):
        calendar.book(
            service_id=haircut.id, starts_at=now, customer_name="Ada", phone="+447700900123"
        )


def test_a_slot_in_the_future_can_still_be_booked(session, calendar_settings, open_weekdays, haircut) -> None:
    calendar = _service(session, calendar_settings, SUNDAY_NOON)

    appointment = calendar.book(
        service_id=haircut.id,
        starts_at=datetime(2026, 3, 2, 10, 0, tzinfo=UTC),
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )

    assert appointment.starts_at == datetime(2026, 3, 2, 10, 0, tzinfo=UTC)


def test_a_booking_cannot_be_moved_into_the_past(session, calendar_settings, open_weekdays, haircut) -> None:
    calendar = _service(session, calendar_settings, SUNDAY_NOON)
    appointment = calendar.book(
        service_id=haircut.id,
        starts_at=datetime(2026, 3, 2, 14, 0, tzinfo=UTC),
        customer_name="Ada",
        phone="+447700900123",
    )
    later = _service(session, calendar_settings, datetime(2026, 3, 2, 12, 0, tzinfo=UTC))

    with pytest.raises(SlotInThePast):
        later.reschedule(appointment.id, datetime(2026, 3, 2, 9, 0, tzinfo=UTC))

    assert appointment.starts_at == datetime(2026, 3, 2, 14, 0, tzinfo=UTC)


def test_today_offers_only_what_is_still_ahead(session, calendar_settings, open_weekdays, haircut) -> None:
    from datetime import date

    calendar = _service(session, calendar_settings, datetime(2026, 3, 2, 12, 10, tzinfo=UTC))

    slots = calendar.available_slots(haircut.id, date(2026, 3, 2))

    assert slots, "the afternoon is still open"
    assert all(slot.starts_at > datetime(2026, 3, 2, 12, 10, tzinfo=UTC) for slot in slots)
    assert slots[0].starts_at == datetime(2026, 3, 2, 12, 15, tzinfo=UTC)


def test_a_day_gone_by_offers_nothing(session, calendar_settings, open_weekdays, haircut) -> None:
    from datetime import date

    calendar = _service(session, calendar_settings, datetime(2026, 3, 3, 8, 0, tzinfo=UTC))

    assert calendar.available_slots(haircut.id, date(2026, 3, 2)) == []


def test_is_available_is_false_for_a_past_time(session, calendar_settings, open_weekdays, haircut) -> None:
    calendar = _service(session, calendar_settings, datetime(2026, 3, 2, 12, 0, tzinfo=UTC))

    assert not calendar.is_available(haircut.id, datetime(2026, 3, 2, 10, 0, tzinfo=UTC))
    assert calendar.is_available(haircut.id, datetime(2026, 3, 2, 15, 0, tzinfo=UTC))


def test_the_default_clock_is_the_system_clock_looked_up_at_call_time() -> None:
    """The suite replaces `app.clock.system_clock`; the calendar must see that, not a copy."""
    import app.clock

    assert app.clock.system_clock() == SUITE_NOW


def test_a_fixed_clock_needs_an_aware_instant() -> None:
    with pytest.raises(ValueError):
        fixed_clock(datetime(2026, 3, 1, 12, 0))


def test_the_real_system_clock_is_aware_and_current(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()
    now = system_clock()

    assert now.tzinfo is not None
    assert abs(now - datetime.now(UTC)) < timedelta(seconds=5)
