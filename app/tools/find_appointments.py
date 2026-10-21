"""`find_appointments` — look up what a caller already has booked.

Reschedule and cancel act on an appointment identifier, and the only identifiers a model ever sees come from this tool
or from a booking made earlier in the same call. This tool is therefore the identity check: it returns an appointment
only when the caller's **name and telephone number both match** the booking, and only one that is still ahead of us.
The matching rules live with the calendar (`app/calendar/lookup.py`); this module normalises the arguments, delegates
and describes the outcome, like every other tool.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from app.calendar.lookup import MINIMUM_DIGITS, phone_key
from app.tools.base import ToolArgumentError, ToolContext, ToolResult, require_text


def find_appointments(
    context: ToolContext, *, customer_name: str, phone: str
) -> ToolResult:
    """The caller's upcoming booked appointments, earliest first. An empty list is a successful answer."""
    try:
        name = require_text(customer_name, "customer_name")
        number = require_text(phone, "phone")
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))

    if len(phone_key(number)) < MINIMUM_DIGITS:
        return ToolResult.failed(
            f"phone must be a telephone number with at least {MINIMUM_DIGITS} digits."
        )

    rows = context.calendar().find_upcoming(customer_name=name, phone=number)
    timezone = context.timezone()
    return ToolResult.ok(
        found=len(rows),
        appointments=[
            {
                "appointment_id": str(row.id),
                "service": row.service.name,
                "starts_at": _local(row.starts_at, timezone),
                "ends_at": _local(row.ends_at, timezone),
            }
            for row in rows
        ],
    )


def _local(moment: datetime, timezone: ZoneInfo) -> str:
    return moment.astimezone(timezone).isoformat()
