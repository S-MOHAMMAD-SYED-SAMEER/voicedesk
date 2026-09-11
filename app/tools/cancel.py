"""`cancel` — give a slot back.

The row is not deleted. The exclusion constraint is partial, so marking an
appointment cancelled is what frees its interval, and the call that booked it
still points at something afterwards.
"""

from uuid import UUID

from app.calendar import CalendarError
from app.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolResult,
    parse_identifier,
)


def cancel(context: ToolContext, *, appointment_id: UUID | str) -> ToolResult:
    """Cancel an existing appointment."""
    try:
        identifier = parse_identifier(appointment_id, "appointment_id")
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))

    try:
        appointment = context.calendar().cancel(identifier)
    except CalendarError as exc:
        return ToolResult.failed(str(exc))

    timezone = context.timezone()
    return ToolResult.ok(
        appointment_id=str(appointment.id),
        customer_name=appointment.customer_name,
        starts_at=appointment.starts_at.astimezone(timezone).isoformat(),
        status=str(appointment.status),
    )
