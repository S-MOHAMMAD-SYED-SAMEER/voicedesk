"""`take_message` — the graceful way to fail.

When the caller wants something the calendar cannot express, the receptionist takes a message rather than improvising.
The message is **stored** as a staff request (`app/models/staff_request.py`) and shown in the staff view, so what the
caller is told — that the team will call back — has something behind it. It is also kept in the `tool_calls` row of
the turn, as every tool call is.

All three fields are required: a message nobody can be called back about is not a message.
"""

from app.models import StaffRequest, StaffRequestKind
from app.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolResult,
    require_text,
)


def take_message(
    context: ToolContext, *, caller_name: str, phone: str, message: str
) -> ToolResult:
    """Record a message for the team to deal with later."""
    try:
        name = require_text(caller_name, "caller_name")
        number = require_text(phone, "phone")
        body = require_text(message, "message")
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))

    request = StaffRequest(
        call_id=context.call_id,
        kind=StaffRequestKind.MESSAGE,
        caller_name=name,
        phone=number,
        body=body,
    )
    context.session.add(request)
    context.session.commit()
    return ToolResult.ok(
        recorded=True,
        caller_name=name,
        phone=number,
        message=body,
        call_id=str(context.call_id) if context.call_id else None,
        staff_request_id=str(request.id),
        what_to_tell_the_caller=(
            "The message has been passed to the team and they will call back. "
            "Do not say anyone is on the line or is being put through."
        ),
    )
