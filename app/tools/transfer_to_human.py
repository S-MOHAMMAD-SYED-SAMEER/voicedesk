"""`transfer_to_human` — know when to stop.

The specification cares about "escalation precision and recall", so what matters is that the decision to escalate is
recorded, with the reason that prompted it, in a form the evaluation harness can count.

**There is no live transfer.** Nothing in this application can connect a caller to a person, and a receptionist that
said it was putting someone through would be lying. So this tool stores a *request that the team call the caller back*
(`app/models/staff_request.py`, shown in the staff view) and tells the model, in the result, what it may truthfully say.
The caller's number is whatever the carrier reported for the call; the model is never asked to guess one.
"""

from app.models import Call, StaffRequest, StaffRequestKind
from app.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolResult,
    require_text,
)


def transfer_to_human(context: ToolContext, *, reason: str) -> ToolResult:
    """Ask the team to call the caller back, recording why.

    The reason is required. An escalation without one cannot be judged right
    or wrong afterwards, which is exactly what the metric needs to do.
    """
    try:
        why = require_text(reason, "reason")
    except ToolArgumentError as exc:
        return ToolResult.failed(str(exc))

    phone = None
    if context.call_id is not None:
        call = context.session.get(Call, context.call_id)
        phone = call.from_number if call is not None else None

    request = StaffRequest(
        call_id=context.call_id,
        kind=StaffRequestKind.TRANSFER,
        phone=phone,
        body=why,
    )
    context.session.add(request)
    context.session.commit()
    return ToolResult.ok(
        escalated=True,
        recorded=True,
        live_transfer_available=False,
        reason=why,
        call_id=str(context.call_id) if context.call_id else None,
        staff_request_id=str(request.id),
        what_to_tell_the_caller=(
            "A person cannot take this call. Say that you have passed it to the "
            "team and that they will call back. Never say you are transferring "
            "or putting the caller through."
        ),
    )
