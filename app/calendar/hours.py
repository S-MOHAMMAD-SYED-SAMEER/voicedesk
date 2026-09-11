"""Business hours: turning wall-clock opening times into real instants.

`business_hours` stores wall-clock times — "we open at nine" is not a moment.
Appointments are instants. This module is the only place the two meet, and it
does so through one configured business timezone (see `Settings`); the
specification defines no timezone and VoiceDesk serves a single business.

A known limitation: an opening period on a daylight-saving transition day is
resolved by `zoneinfo`'s default fold handling, not by any rule of our own.
With the default UTC there are no transitions.
"""

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import BusinessHours


@dataclass(frozen=True)
class OpeningPeriod:
    """One opening period on one calendar day, as instants."""

    opens_at: datetime
    closes_at: datetime


def at(day: date, wall_clock: time, timezone: ZoneInfo) -> datetime:
    """The instant at which `wall_clock` happens on `day` in `timezone`."""
    return datetime.combine(day, wall_clock).replace(tzinfo=timezone)


def periods_for(session: Session, day: date, timezone: ZoneInfo) -> list[OpeningPeriod]:
    """Every opening period on `day`, earliest first.

    A weekday may have more than one row — a lunch break is two periods — so
    this returns a list rather than a single pair.
    """
    rows = (
        session.execute(
            select(BusinessHours)
            .where(BusinessHours.weekday == day.weekday())
            .order_by(BusinessHours.opens_at)
        )
        .scalars()
        .all()
    )
    return [at_period(day, row, timezone) for row in rows]


def at_period(day: date, row: BusinessHours, timezone: ZoneInfo) -> OpeningPeriod:
    return OpeningPeriod(
        opens_at=at(day, row.opens_at, timezone),
        closes_at=at(day, row.closes_at, timezone),
    )


def fits_in_one_period(
    session: Session, starts_at: datetime, ends_at: datetime, timezone: ZoneInfo
) -> bool:
    """Does `[starts_at, ends_at)` sit entirely inside one opening period?

    One period, not several: an appointment may not run through a lunch break
    even if both sides of it are open. Starting exactly at opening is inside;
    ending exactly at closing is inside; starting exactly at closing is not,
    because nothing fits after it.

    An interval that crosses local midnight belongs to two days and cannot sit
    inside any single period, so it is refused here rather than silently split.
    """
    local_start = starts_at.astimezone(timezone)
    local_end = ends_at.astimezone(timezone)
    if local_start.date() != local_end.date():
        return False

    return any(
        period.opens_at <= starts_at and ends_at <= period.closes_at
        for period in periods_for(session, local_start.date(), timezone)
    )
