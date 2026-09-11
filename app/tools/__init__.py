"""The tool layer: the only things a model is allowed to do.

Six tools, one module each, every one a plain function that takes a
`ToolContext` and keyword arguments and returns a `ToolResult`. They are
callable from a test with a database session and nothing else — no audio, no
telephony, no model.

Each tool is deliberately thin. Availability, business hours, durations and
conflict handling all live in `app/calendar/` and are reached through
`CalendarService`; a tool normalises its arguments, delegates, and describes
what happened. Nothing here recomputes what the calendar already knows,
because two implementations of "is this free?" are one implementation too
many.

Failures are returned, not raised. A model that asks for a slot that has just
gone needs to be told so in a way it can act on; an exception would end the
turn instead.

`TOOLS` maps the six names in the specification to those functions. The
dialogue layer in milestone 4 builds its tool definitions from this registry
rather than keeping a second list that can drift out of step.
"""

from collections.abc import Callable

from app.tools.base import (
    ServiceLookupError,
    ToolArgumentError,
    ToolContext,
    ToolResult,
)
from app.tools.book_appointment import book_appointment
from app.tools.cancel import cancel
from app.tools.check_availability import check_availability
from app.tools.reschedule import reschedule
from app.tools.take_message import take_message
from app.tools.transfer_to_human import transfer_to_human

TOOLS: dict[str, Callable[..., ToolResult]] = {
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "reschedule": reschedule,
    "cancel": cancel,
    "take_message": take_message,
    "transfer_to_human": transfer_to_human,
}


class UnknownTool(Exception):
    """A name that is not one of the six tools."""


def get_tool(name: str) -> Callable[..., ToolResult]:
    """Look up a tool by the name a model would use.

    Unknown names raise rather than return a failed result: a model asking for
    a tool that does not exist is a dialogue-layer problem, not something to
    report back to the caller as a business outcome.
    """
    try:
        return TOOLS[name]
    except KeyError:
        raise UnknownTool(
            f"{name!r} is not a tool. Available: {', '.join(sorted(TOOLS))}."
        ) from None


__all__ = [
    "TOOLS",
    "ServiceLookupError",
    "ToolArgumentError",
    "ToolContext",
    "ToolResult",
    "UnknownTool",
    "book_appointment",
    "cancel",
    "check_availability",
    "get_tool",
    "reschedule",
    "take_message",
    "transfer_to_human",
]
