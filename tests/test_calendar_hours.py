"""Business hours: wall-clock configuration meeting real instants."""

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from app.calendar import hours
from app.models import BusinessHours

MONDAY = datetime(2026, 3, 2, tzinfo=UTC).date()
SATURDAY = datetime(2026, 3, 7, tzinfo=UTC).date()
UTC_ZONE = ZoneInfo("UTC")


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC)


def _fits(session: Session, start: datetime, minutes: int) -> bool:
    return hours.fits_in_one_period(
        session, start, start + timedelta(minutes=minutes), UTC_ZONE
    )


def test_the_day_is_read_from_the_database(session: Session, open_weekdays) -> None:
    periods = hours.periods_for(session, MONDAY, UTC_ZONE)

    assert len(periods) == 1
    assert periods[0].opens_at == _at(9)
    assert periods[0].closes_at == _at(17)


def test_a_closed_day_has_no_periods(session: Session, open_weekdays) -> None:
    assert hours.periods_for(session, SATURDAY, UTC_ZONE) == []


def test_inside_hours_fits(session: Session, open_weekdays) -> None:
    assert _fits(session, _at(11), 30)


def test_exactly_at_opening_fits(session: Session, open_weekdays) -> None:
    assert _fits(session, _at(9), 30)


def test_ending_exactly_at_closing_fits(session: Session, open_weekdays) -> None:
    assert _fits(session, _at(16, 30), 30)


def test_starting_exactly_at_closing_does_not_fit(
    session: Session, open_weekdays
) -> None:
    """Nothing fits after closing time, not even a zero-length wish."""
    assert not _fits(session, _at(17), 30)


def test_before_opening_does_not_fit(session: Session, open_weekdays) -> None:
    assert not _fits(session, _at(8, 45), 30)


def test_overrunning_closing_does_not_fit(session: Session, open_weekdays) -> None:
    assert not _fits(session, _at(16, 45), 30)


def test_after_closing_does_not_fit(session: Session, open_weekdays) -> None:
    assert not _fits(session, _at(18), 30)


def test_a_closed_day_fits_nothing(session: Session, open_weekdays) -> None:
    saturday = datetime(2026, 3, 7, 11, tzinfo=UTC)

    assert not hours.fits_in_one_period(
        session, saturday, saturday + timedelta(minutes=30), UTC_ZONE
    )


def test_no_configured_hours_at_all_fits_nothing(session: Session) -> None:
    assert not _fits(session, _at(11), 30)


# --- several periods in one day -------------------------------------------


@pytest.fixture
def open_with_lunch_break(session: Session) -> None:
    session.add_all(
        [
            BusinessHours(weekday=0, opens_at=time(9), closes_at=time(12, 30)),
            BusinessHours(weekday=0, opens_at=time(13, 30), closes_at=time(17)),
        ]
    )
    session.commit()


def test_both_periods_are_returned_in_order(
    session: Session, open_with_lunch_break
) -> None:
    periods = hours.periods_for(session, MONDAY, UTC_ZONE)

    assert [period.opens_at for period in periods] == [_at(9), _at(13, 30)]


def test_either_side_of_the_break_fits(
    session: Session, open_with_lunch_break
) -> None:
    assert _fits(session, _at(11), 30)
    assert _fits(session, _at(14), 30)


def test_the_break_itself_does_not_fit(
    session: Session, open_with_lunch_break
) -> None:
    assert not _fits(session, _at(12, 45), 30)


def test_an_appointment_may_not_run_through_the_break(
    session: Session, open_with_lunch_break
) -> None:
    """12:00–14:00 is inside opening hours twice over, and still refused."""
    assert not _fits(session, _at(12), 120)


# --- timezone semantics ----------------------------------------------------


def test_hours_are_read_in_the_configured_timezone(session: Session) -> None:
    """09:00 in New York is 14:00 UTC, and the stored 9 means the former."""
    session.add(BusinessHours(weekday=0, opens_at=time(9), closes_at=time(17)))
    session.commit()
    new_york = ZoneInfo("America/New_York")

    nine_local = datetime(2026, 3, 2, 14, tzinfo=UTC)  # 09:00 EST
    eight_local = datetime(2026, 3, 2, 13, tzinfo=UTC)  # 08:00 EST

    assert hours.fits_in_one_period(
        session, nine_local, nine_local + timedelta(minutes=30), new_york
    )
    assert not hours.fits_in_one_period(
        session, eight_local, eight_local + timedelta(minutes=30), new_york
    )


def test_an_interval_crossing_local_midnight_is_refused(session: Session) -> None:
    """It belongs to two days, so it can sit inside no single period."""
    session.add_all(
        [
            BusinessHours(weekday=0, opens_at=time(9), closes_at=time(23, 59)),
            BusinessHours(weekday=1, opens_at=time(0), closes_at=time(17)),
        ]
    )
    session.commit()
    late = datetime(2026, 3, 2, 23, 30, tzinfo=UTC)

    assert not hours.fits_in_one_period(
        session, late, late + timedelta(minutes=60), UTC_ZONE
    )
