"""The tool definitions the model is given."""

import inspect
import json

import pytest

from app.dialogue.definitions import TOOL_ORDER, tool_definitions
from app.providers.llm import ToolDefinition
from app.tools import TOOLS


def test_exactly_the_six_registered_tools_are_described() -> None:
    definitions = tool_definitions()

    assert len(definitions) == 6
    assert {definition.name for definition in definitions} == set(TOOLS)


def test_the_order_is_stable() -> None:
    """Stable request bytes: a varying tool list would defeat caching later."""
    assert [definition.name for definition in tool_definitions()] == list(TOOL_ORDER)
    assert tool_definitions() == tool_definitions()


def test_every_schema_matches_its_tool_signature() -> None:
    """The registry is authoritative; a schema that drifts is a failure."""
    for definition in tool_definitions():
        expected = {
            parameter.name
            for parameter in inspect.signature(TOOLS[definition.name]).parameters.values()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert set(definition.input_schema["properties"]) == expected, definition.name


def test_every_argument_is_required() -> None:
    for definition in tool_definitions():
        schema = definition.input_schema
        assert set(schema["required"]) == set(schema["properties"]), definition.name


def test_no_schema_accepts_extra_properties() -> None:
    """Strict tool use needs closed objects."""
    for definition in tool_definitions():
        assert definition.input_schema["additionalProperties"] is False
        assert definition.input_schema["type"] == "object"


def test_every_definition_is_strict() -> None:
    assert all(definition.strict for definition in tool_definitions())


def test_no_tool_exposes_a_call_id() -> None:
    """The call is ambient. A model may not attribute work to another call."""
    for definition in tool_definitions():
        assert "call_id" not in definition.input_schema["properties"], definition.name


def test_every_property_is_a_described_string() -> None:
    for definition in tool_definitions():
        for name, spec in definition.input_schema["properties"].items():
            assert spec["type"] == "string", f"{definition.name}.{name}"
            assert spec["description"].strip(), f"{definition.name}.{name}"


def test_definitions_are_json_serialisable() -> None:
    payload = [
        {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
            "strict": definition.strict,
        }
        for definition in tool_definitions()
    ]
    assert json.loads(json.dumps(payload)) == payload


def test_the_date_and_time_formats_are_spelled_out() -> None:
    """A model guessing at a format is a whole class of avoidable failure."""
    schemas = {
        definition.name: definition.input_schema["properties"]
        for definition in tool_definitions()
    }
    assert "ISO date" in schemas["check_availability"]["day"]["description"]
    assert "ISO-8601" in schemas["book_appointment"]["starts_at"]["description"]
    assert "ISO-8601" in schemas["reschedule"]["new_starts_at"]["description"]


def test_no_schema_restates_calendar_business_logic() -> None:
    """Durations, opening hours and availability live in the calendar."""
    text = json.dumps(
        [
            {"d": definition.description, "s": definition.input_schema}
            for definition in tool_definitions()
        ]
    ).lower()
    for forbidden in ("duration", "minutes", "opening hours", "business hours", "09:00"):
        assert forbidden not in text, forbidden


def test_no_schema_enumerates_the_services() -> None:
    """Services are data. They reach the model through the system prompt."""
    text = json.dumps([d.input_schema for d in tool_definitions()])
    assert "Haircut" not in text


def test_a_registered_tool_with_no_schema_is_refused(monkeypatch) -> None:
    """Adding a tool to the registry and forgetting its schema must fail loudly."""
    import app.dialogue.definitions as definitions

    monkeypatch.setitem(definitions.TOOLS, "order_a_taxi", lambda context: None)

    with pytest.raises(RuntimeError, match="order_a_taxi"):
        tool_definitions()


def test_a_schema_for_an_unregistered_tool_is_refused(monkeypatch) -> None:
    import app.dialogue.definitions as definitions

    monkeypatch.setitem(definitions._TOOLS, "order_a_taxi", ("Taxi.", {}))

    with pytest.raises(RuntimeError, match="not registered"):
        tool_definitions()


def test_a_schema_that_drifts_from_its_signature_is_refused(monkeypatch) -> None:
    import app.dialogue.definitions as definitions

    monkeypatch.setitem(
        definitions._TOOLS, "cancel", ("Cancel it.", {"booking_ref": {"type": "string"}})
    )

    with pytest.raises(RuntimeError, match="does not match its signature"):
        tool_definitions()


def test_definitions_are_the_provider_neutral_dataclass() -> None:
    assert all(
        isinstance(definition, ToolDefinition) for definition in tool_definitions()
    )
