"""`book_appointment` — turn an agreed time into a commitment.

The tool does not check availability first. `CalendarService.book` deliberately
writes and lets the database's exclusion constraint refuse an overlap, because
any check taken beforehand can be stale by the time the write lands. A refusal
arrives here as `SlotUnavailable` and leaves as an ordinary failed result: the
caller is told the slot has gone and can ask for availability again.
"""

from datetime import datetime

from app.calendar import CalendarError, SlotUnavailable
from app.tools.base import (
    ServiceLookupError,
    ToolArgumentError,
    ToolContext,
    ToolResult,
    parse_instant,
    require_text,
    resolve_service,
)


def book_appointment(
    context: ToolContext,
    *,
    service_name: str,
    starts_at: datetime | str,
    customer_name: str,
    phone: str,
) -> ToolResult:
    """Book one appointment, or say why it could not be booked.

    `call_id` comes from the context rather than the arguments, so a booking
    is always attributed to the call that is actually happening.
    """
    try:
        name = require_text(customer_name, "customer_name")
        number = require_text(phone, "phone")
        when = parse_instant(starts_at, context.timezone())
        service = resolve_service(context.session, service_name)
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))
    except ServiceLookupError as exc:
        return ToolResult.failed(str(exc), services_offered=exc.offered)

    try:
        appointment = context.calendar().book(
            service_id=service.id,
            starts_at=when,
            customer_name=name,
            phone=number,
            call_id=context.call_id,
        )
    except SlotUnavailable as exc:
        # Expected under contention, not an error in the caller's request.
        return ToolResult.failed(str(exc), slot_taken=True)
    except CalendarError as exc:
        return ToolResult.failed(str(exc))

    timezone = context.timezone()
    return ToolResult.ok(
        appointment_id=str(appointment.id),
        service=service.name,
        customer_name=appointment.customer_name,
        phone=appointment.phone,
        starts_at=appointment.starts_at.astimezone(timezone).isoformat(),
        ends_at=appointment.ends_at.astimezone(timezone).isoformat(),
        status=str(appointment.status),
    )
