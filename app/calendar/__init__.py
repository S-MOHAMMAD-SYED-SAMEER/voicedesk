"""Calendar core: deterministic availability and database-backed booking.

The public surface is `CalendarService` and the errors it raises. Later
milestones reach the calendar only through this package — the dialogue layer
must never work out availability for itself.
"""

from app.calendar.availability import Slot
from app.calendar.errors import (
    AppointmentAlreadyCancelled,
    AppointmentNotFound,
    AppointmentNotReschedulable,
    CalendarError,
    InvalidInterval,
    InvalidServiceDuration,
    OutsideBusinessHours,
    ServiceNotFound,
    SlotUnavailable,
)
from app.calendar.hours import OpeningPeriod
from app.calendar.service import CalendarService

__all__ = [
    "AppointmentAlreadyCancelled",
    "AppointmentNotFound",
    "AppointmentNotReschedulable",
    "CalendarError",
    "CalendarService",
    "InvalidInterval",
    "InvalidServiceDuration",
    "OpeningPeriod",
    "OutsideBusinessHours",
    "ServiceNotFound",
    "Slot",
    "SlotUnavailable",
]
