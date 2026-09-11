"""The registry: six tools, named as the specification names them."""

import inspect

import pytest

from app.tools import TOOLS, ToolContext, ToolResult, UnknownTool, get_tool

SPECIFIED = {
    "check_availability",
    "book_appointment",
    "reschedule",
    "cancel",
    "take_message",
    "transfer_to_human",
}


def test_the_registry_holds_exactly_the_specified_tools() -> None:
    assert set(TOOLS) == SPECIFIED


def test_every_tool_is_reachable_by_name() -> None:
    for name in SPECIFIED:
        assert get_tool(name) is TOOLS[name]


def test_an_unknown_name_raises_rather_than_returning_a_failure() -> None:
    """A tool that does not exist is a dialogue-layer bug, not a business one."""
    with pytest.raises(UnknownTool) as caught:
        get_tool("order_a_taxi")

    assert "check_availability" in str(caught.value)


def test_every_tool_takes_a_context_first_and_keywords_after() -> None:
    """One shape, so milestone 4 can dispatch without special cases."""
    for name, tool in TOOLS.items():
        parameters = list(inspect.signature(tool).parameters.values())
        assert parameters[0].name == "context", name
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in parameters[1:]
        ), name


def test_no_tool_takes_a_call_id_argument() -> None:
    """The call is ambient, so a model cannot attribute work to another call."""
    for name, tool in TOOLS.items():
        assert "call_id" not in inspect.signature(tool).parameters, name


def test_every_tool_returns_a_tool_result(
    tools: ToolContext, open_weekdays, haircut
) -> None:
    """Including when the arguments are nonsense — failures are returned."""
    for name, tool in TOOLS.items():
        arguments = {
            parameter: ""
            for parameter in inspect.signature(tool).parameters
            if parameter != "context"
        }
        result = tool(tools, **arguments)
        assert isinstance(result, ToolResult), name
        assert result.success is False, name
        assert result.error, name


def test_a_result_serialises_to_plain_json_shaped_values() -> None:
    """`tool_calls.result` is JSONB; milestone 4 writes this straight in."""
    import json

    result = ToolResult.ok(appointment_id="abc", slots=[])
    assert json.loads(json.dumps(result.as_dict())) == {
        "success": True,
        "data": {"appointment_id": "abc", "slots": []},
        "error": None,
    }


def test_a_failed_result_carries_its_error() -> None:
    result = ToolResult.failed("no such service", services_offered=["Haircut"])

    assert result.as_dict() == {
        "success": False,
        "data": {"services_offered": ["Haircut"]},
        "error": "no such service",
    }
