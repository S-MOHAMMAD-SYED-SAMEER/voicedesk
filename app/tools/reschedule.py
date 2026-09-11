"""`reschedule` — move a booking without losing it.

The appointment keeps its identity: same id, same originating call, new time.
The service is not an argument, because rescheduling is not re-choosing what
was booked — `CalendarService` reads the service from the appointment itself.
"""

from datetime import datetime
from uuid import UUID

from app.calendar import CalendarError, SlotUnavailable
from app.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolResult,
    parse_identifier,
    parse_instant,
)


def reschedule(
    context: ToolContext,
    *,
    appointment_id: UUID | str,
    new_starts_at: datetime | str,
) -> ToolResult:
    """Move an existing appointment to a new start time."""
    try:
        identifier = parse_identifier(appointment_id, "appointment_id")
        when = parse_instant(new_starts_at, context.timezone(), "new_starts_at")
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))

    try:
        appointment = context.calendar().reschedule(identifier, when)
    except SlotUnavailable as exc:
        return ToolResult.failed(str(exc), slot_taken=True)
    except CalendarError as exc:
        return ToolResult.failed(str(exc))

    timezone = context.timezone()
    return ToolResult.ok(
        appointment_id=str(appointment.id),
        customer_name=appointment.customer_name,
        starts_at=appointment.starts_at.astimezone(timezone).isoformat(),
        ends_at=appointment.ends_at.astimezone(timezone).isoformat(),
        status=str(appointment.status),
    )
