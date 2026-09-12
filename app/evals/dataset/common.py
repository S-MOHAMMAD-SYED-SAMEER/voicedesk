"""The pieces every scenario is built from.

One business, one fixed week, one set of invented callers. Scenarios differ in
what is already booked and in what is said, not in the ground they stand on —
so a failure points at behaviour rather than at a fixture.

**Fixed dates, on purpose.** Nothing in `app/calendar/` or `app/tools/` reads
the wall clock: availability is a pure function of `business_hours`, the
service, the appointments already made and the day that was asked for. A date
written here therefore means the same thing today and in five years, with no
clock to freeze and no dependency to add.

**Synthetic people.** Every name is invented and every number is from the UK's
reserved drama range, which is never allocated to a real subscriber. No real
personal information appears anywhere in this dataset.

A reply that names a time uses a numeral — `10:00`, not "ten" — so that the
undeclared-claim detector in `checks.py` can actually see it. A reply that
mentions a time the calendar did **not** offer says so in words instead, for
the same reason: the detector is deliberately unable to tell "10:00 is free"
from "10:00 is taken", so the dataset never asks it to.
"""

from app.evals.scenario import HoursSpec, ServiceSpec, World, weekdays

# A Monday. The week the whole dataset happens in.
MONDAY = "2026-03-02"
TUESDAY = "2026-03-03"
SATURDAY = "2026-03-07"


def at(clock: str, day: str = MONDAY) -> str:
    """An instant on a day of the fixed week, written the way a tool wants it."""
    return f"{day}T{clock}:00+00:00"


NINE = at("09:00")
NINE_THIRTY = at("09:30")
TEN = at("10:00")
ELEVEN = at("11:00")
TWO = at("14:00")
THREE = at("15:00")

HAIRCUT = "Haircut"
BEARD_TRIM = "Beard trim"

# Invented callers. The numbers are in Ofcom's reserved drama range.
ADA = ("Ada Lovelace", "+447700900123")
BEA = ("Bea Bramble", "+447700900456")
CAI = ("Cai Rivers", "+447700900789")


def salon(*services: ServiceSpec, hours: tuple[HoursSpec, ...] | None = None) -> World:
    """The business: whatever services are named, open on weekdays."""
    return World(
        services=services or (ServiceSpec(HAIRCUT, 30, "sam"),),
        hours=hours if hours is not None else weekdays(),
    )


def haircut_only() -> World:
    """One 30-minute service, performed by one person."""
    return salon(ServiceSpec(HAIRCUT, 30, "sam"))


__all__ = [
    "ADA",
    "BEA",
    "BEARD_TRIM",
    "CAI",
    "ELEVEN",
    "HAIRCUT",
    "MONDAY",
    "NINE",
    "NINE_THIRTY",
    "SATURDAY",
    "TEN",
    "THREE",
    "TUESDAY",
    "TWO",
    "at",
    "haircut_only",
    "salon",
]
