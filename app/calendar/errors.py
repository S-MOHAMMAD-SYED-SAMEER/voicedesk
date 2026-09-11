"""Domain errors the calendar raises.

Each one names a business situation, not a database or HTTP condition. The
tool layer in a later milestone turns these into something a caller can be
told; the calendar itself has no opinion about transport.
"""


class CalendarError(Exception):
    """Base for everything this package raises."""


class ServiceNotFound(CalendarError):
    """No bookable service with that id.

    Also raised for a service that exists but is not `active`: an inactive
    service is not bookable, and the message says which case it was.
    """


class AppointmentNotFound(CalendarError):
    """No appointment with that id."""


class InvalidServiceDuration(CalendarError):
    """The service's `duration_minutes` cannot produce a real interval."""


class InvalidInterval(CalendarError):
    """The requested interval is not a usable one.

    A naive datetime, or a start that is not before its end.
    """


class OutsideBusinessHours(CalendarError):
    """The interval does not fit inside a single configured opening period."""


class SlotUnavailable(CalendarError):
    """The staff member is already booked across part of that interval.

    Raised when the database's exclusion constraint refuses the write, which
    is the only authoritative answer — an availability check taken earlier can
    always have gone stale.
    """


class AppointmentAlreadyCancelled(CalendarError):
    """The appointment is already cancelled."""


class AppointmentNotReschedulable(CalendarError):
    """The appointment is not in a state that can be moved."""
