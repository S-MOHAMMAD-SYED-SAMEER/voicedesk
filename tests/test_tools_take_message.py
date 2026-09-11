"""`take_message` — the graceful exit when the calendar cannot help."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Appointment, ToolCall
from app.tools import ToolContext, take_message


def test_a_message_comes_back_whole(tools: ToolContext) -> None:
    result = take_message(
        tools,
        caller_name="Ada Lovelace",
        phone="+447700900123",
        message="Please call me about the invoice.",
    )

    assert result.success, result.error
    assert result.data["caller_name"] == "Ada Lovelace"
    assert result.data["phone"] == "+447700900123"
    assert result.data["message"] == "Please call me about the invoice."


def test_the_message_carries_the_call_it_belongs_to(
    session: Session, calendar_settings
) -> None:
    call_id = uuid.uuid4()
    context = ToolContext(
        session=session, settings=calendar_settings, call_id=call_id
    )

    result = take_message(
        context, caller_name="Ada", phone="+447700900123", message="Call back."
    )

    assert result.data["call_id"] == str(call_id)


def test_without_a_call_the_message_says_so(tools: ToolContext) -> None:
    result = take_message(
        tools, caller_name="Ada", phone="+447700900123", message="Call back."
    )

    assert result.data["call_id"] is None


def test_whitespace_is_trimmed(tools: ToolContext) -> None:
    result = take_message(
        tools,
        caller_name="  Ada Lovelace  ",
        phone=" +447700900123 ",
        message="  Call back.  ",
    )

    assert result.data["caller_name"] == "Ada Lovelace"
    assert result.data["message"] == "Call back."


def test_a_message_with_no_number_to_call_back_is_refused(
    tools: ToolContext,
) -> None:
    result = take_message(tools, caller_name="Ada", phone="  ", message="Call back.")

    assert not result.success
    assert "phone is required" in result.error


def test_an_empty_message_is_refused(tools: ToolContext) -> None:
    result = take_message(tools, caller_name="Ada", phone="+447700900123", message="")

    assert not result.success
    assert "message is required" in result.error


def test_a_missing_caller_name_is_refused(tools: ToolContext) -> None:
    result = take_message(
        tools, caller_name="", phone="+447700900123", message="Call back."
    )

    assert not result.success
    assert "caller_name is required" in result.error


def test_taking_a_message_writes_nothing(
    session: Session, tools: ToolContext
) -> None:
    """Persistence is milestone 4's: `tool_calls.turn_id` needs a turn."""
    take_message(
        tools, caller_name="Ada", phone="+447700900123", message="Call back."
    )

    assert session.execute(select(ToolCall)).first() is None
    assert session.execute(select(Appointment)).first() is None
