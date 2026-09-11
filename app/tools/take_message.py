"""`take_message` — the graceful way to fail.

When the caller wants something the calendar cannot express, the receptionist
takes a message rather than improvising. The tool does nothing clever: it
checks that the three things a message needs are actually present and hands
them back.

It writes nothing. `tool_calls.turn_id` is required, and turns are the
dialogue layer's to create, so the message travels in `ToolResult.data` and
milestone 4 persists it along with every other tool call. Duplicating it into
a table of its own now would give the same message two homes.
"""

from app.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolResult,
    require_text,
)


def take_message(
    context: ToolContext, *, caller_name: str, phone: str, message: str
) -> ToolResult:
    """Record a message for a human to deal with later.

    All three fields are required: a message nobody can be called back about
    is not a message.
    """
    try:
        name = require_text(caller_name, "caller_name")
        number = require_text(phone, "phone")
        body = require_text(message, "message")
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))

    return ToolResult.ok(
        caller_name=name,
        phone=number,
        message=body,
        call_id=str(context.call_id) if context.call_id else None,
    )
