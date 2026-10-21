"""What time it is, in one place that a test or an evaluation can replace.

The calendar refuses a time that has already passed and the dialogue layer tells the model today's date, so both need a
clock. They take one as an argument, defaulting to the system's, so nothing below this line reads the wall clock
directly and a scenario can say it is a particular Sunday afternoon.
"""

from collections.abc import Callable
from datetime import UTC, datetime

Clock = Callable[[], datetime]


def system_clock() -> datetime:
    """The current instant, timezone-aware, in UTC."""
    return datetime.now(UTC)


def default_clock() -> Clock:
    """The clock used when none is given: the system's, looked up at call time (so a test can replace it)."""
    return lambda: system_clock()


def fixed_clock(instant: datetime) -> Clock:
    """A clock that always says `instant` (which must be timezone-aware)."""
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("a fixed clock needs a timezone-aware instant")
    return lambda: instant


__all__ = ["Clock", "default_clock", "fixed_clock", "system_clock"]
