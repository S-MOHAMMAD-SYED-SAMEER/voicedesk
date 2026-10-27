"""Overlapping durations: a 30-minute and a 15-minute service for the same stylist, against real PostgreSQL.

The calendar compares whole intervals, not start times. A booking blocks every start whose own interval overlaps it, and
only those: touching at the edge is allowed. No model is involved; these are the rules the real-model scenario
`reschedule_unavailable` relies on.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.calendar import CalendarService, SlotUnavailable
from app.models import Service

MONDAY = datetime(2026, 3, 2, tzinfo=UTC).date()


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC)


@pytest.fixture
def beard_trim(session: Session) -> Service:
    """A 15-minute service by the same stylist as the haircut."""
    service = Service(name="Beard trim", duration_minutes=15, staff_id="sam")
    session.add(service)
    session.commit()
    return service


def _book(calendar: CalendarService, service: Service, start: datetime, name: str = "Bea Bramble"):
    return calendar.book(service_id=service.id, starts_at=start, customer_name=name, phone="+447700900456")


def _starts(calendar: CalendarService, service: Service) -> set[str]:
    return {slot.starts_at.strftime("%H:%M") for slot in calendar.available_slots(service.id, MONDAY)}


def test_a_haircut_blocks_a_beard_trim_that_starts_inside_it(
    calendar: CalendarService, open_weekdays, haircut: Service, beard_trim: Service
) -> None:
    _book(calendar, haircut, _at(14))

    assert calendar.is_available(beard_trim.id, _at(14)) is False
    assert calendar.is_available(beard_trim.id, _at(14, 15)) is False
    with pytest.raises(SlotUnavailable):
        _book(calendar, beard_trim, _at(14, 15), "Cai Rivers")


def test_a_beard_trim_blocks_a_haircut_that_would_run_into_it(
    calendar: CalendarService, open_weekdays, haircut: Service, beard_trim: Service
) -> None:
    """The 14:15 beard trim occupies 14:15-14:30, so a 30-minute haircut from 14:00 (to 14:30) overlaps it; 13:45 (to 14:15) only touches."""
    _book(calendar, beard_trim, _at(14, 15))

    assert calendar.is_available(haircut.id, _at(14)) is False
    assert calendar.is_available(haircut.id, _at(14, 15)) is False
    with pytest.raises(SlotUnavailable):
        _book(calendar, haircut, _at(14), "Cai Rivers")


def test_services_touching_at_the_edge_do_not_overlap(
    calendar: CalendarService, open_weekdays, haircut: Service, beard_trim: Service
) -> None:
    _book(calendar, beard_trim, _at(14, 15))

    assert calendar.is_available(haircut.id, _at(13, 45)) is True
    assert calendar.is_available(haircut.id, _at(14, 30)) is True
    assert calendar.is_available(beard_trim.id, _at(14)) is True
    assert calendar.is_available(beard_trim.id, _at(14, 30)) is True


def test_the_slot_list_leaves_out_exactly_the_overlapping_starts(
    calendar: CalendarService, open_weekdays, haircut: Service, beard_trim: Service
) -> None:
    _book(calendar, beard_trim, _at(14, 15))

    haircuts = _starts(calendar, haircut)
    trims = _starts(calendar, beard_trim)

    assert {"13:30", "13:45", "14:30"} <= haircuts
    assert not haircuts & {"14:00", "14:15"}
    assert {"14:00", "14:30"} <= trims
    assert "14:15" not in trims


def test_a_haircut_moved_onto_a_slot_that_overlaps_a_longer_booking_is_refused_and_stays_put(
    session: Session, calendar: CalendarService, open_weekdays, haircut: Service
) -> None:
    mine = _book(calendar, haircut, _at(10), "Ada Lovelace")
    _book(calendar, haircut, _at(14))

    for start in (_at(13, 45), _at(14), _at(14, 15)):
        with pytest.raises(SlotUnavailable):
            calendar.reschedule(mine.id, start)

    assert calendar.is_available(haircut.id, _at(10)) is False
    session.refresh(mine)
    assert mine.starts_at == _at(10)
