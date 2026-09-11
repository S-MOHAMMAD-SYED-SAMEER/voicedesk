"""`check_availability` — what a caller can actually be offered.

The single most important rule in the specification is that availability is
never invented: "hallucinated-availability rate (must be zero)". This tool is
how that rule is kept. It asks `CalendarService` for the free slots on a day
and reports them verbatim. It does not filter them, round them, describe them
in prose, or offer "the next available" time of its own devising.
"""

from datetime import date

from app.calendar import CalendarError
from app.tools.base import (
    ServiceLookupError,
    ToolArgumentError,
    ToolContext,
    ToolResult,
    parse_day,
    resolve_service,
)


def check_availability(
    context: ToolContext, *, service_name: str, day: date | str
) -> ToolResult:
    """Free start times for a service on one local day.

    Slots come back as ISO-8601 strings in the configured business timezone,
    earliest first. An empty list is a successful answer — "nothing that day"
    is information, not a failure — so a caller can tell it apart from a
    lookup that went wrong.
    """
    try:
        wanted_day = parse_day(day)
        service = resolve_service(context.session, service_name)
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))
    except ServiceLookupError as exc:
        return ToolResult.failed(str(exc), services_offered=exc.offered)

    timezone = context.timezone()
    try:
        slots = context.calendar().available_slots(service.id, wanted_day)
    except CalendarError as exc:
        return ToolResult.failed(str(exc))

    return ToolResult.ok(
        service=service.name,
        day=wanted_day.isoformat(),
        timezone=str(timezone),
        slots=[
            {
                "starts_at": slot.starts_at.astimezone(timezone).isoformat(),
                "ends_at": slot.ends_at.astimezone(timezone).isoformat(),
            }
            for slot in slots
        ],
    )
