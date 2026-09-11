"""`transfer_to_human` — recording an escalation, not performing one."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ToolCall
from app.tools import ToolContext, transfer_to_human


def test_an_escalation_records_its_reason(tools: ToolContext) -> None:
    result = transfer_to_human(tools, reason="Caller is disputing a charge.")

    assert result.success, result.error
    assert result.data["escalated"] is True
    assert result.data["reason"] == "Caller is disputing a charge."


def test_the_escalation_carries_the_call_it_belongs_to(
    session: Session, calendar_settings
) -> None:
    call_id = uuid.uuid4()
    context = ToolContext(
        session=session, settings=calendar_settings, call_id=call_id
    )

    result = transfer_to_human(context, reason="Caller asked for the manager.")

    assert result.data["call_id"] == str(call_id)


def test_an_escalation_without_a_reason_is_refused(tools: ToolContext) -> None:
    """Escalation precision cannot be judged without knowing why."""
    result = transfer_to_human(tools, reason="   ")

    assert not result.success
    assert "reason is required" in result.error


def test_the_reason_is_trimmed(tools: ToolContext) -> None:
    result = transfer_to_human(tools, reason="  Angry caller.  ")

    assert result.data["reason"] == "Angry caller."


def test_escalating_writes_nothing(session: Session, tools: ToolContext) -> None:
    transfer_to_human(tools, reason="Caller asked for the manager.")

    assert session.execute(select(ToolCall)).first() is None
