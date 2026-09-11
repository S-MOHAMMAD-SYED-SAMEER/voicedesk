"""`transfer_to_human` — know when to stop.

The specification cares about "escalation precision and recall", so what
matters here is that the decision to escalate is recorded, with the reason
that prompted it, in a form the evaluation harness can count.

Connecting the call is telephony's job and belongs to the Twilio milestone.
This tool states the intent; nothing in milestones 3 to 5 can move audio, and
pretending otherwise would make an escalation look handled when the caller is
still sitting there.
"""

from app.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolResult,
    require_text,
)


def transfer_to_human(context: ToolContext, *, reason: str) -> ToolResult:
    """Escalate to a human, recording why.

    The reason is required. An escalation without one cannot be judged right
    or wrong afterwards, which is exactly what the metric needs to do.
    """
    try:
        why = require_text(reason, "reason")
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))

    return ToolResult.ok(
        escalated=True,
        reason=why,
        call_id=str(context.call_id) if context.call_id else None,
    )
