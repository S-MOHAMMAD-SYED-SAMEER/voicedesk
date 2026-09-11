"""The six tools, as the model sees them.

These are descriptions of an interface, not a second implementation. They say
what each argument is and how to spell it; they do not say how long a service
takes, when the business opens or what counts as free, because `services`,
`business_hours` and `app/calendar/` already answer those and two answers
would eventually disagree.

`app.tools.TOOLS` stays the authoritative registry of what is callable. This
module is checked against it — and against the real function signatures — so a
tool cannot be added to one and forgotten in the other.
"""

import inspect
from typing import Any

from app.providers.llm import ToolDefinition
from app.tools import TOOLS

# Ordered as the specification lists them, so the request bytes are stable
# from one call to the next.
TOOL_ORDER = (
    "check_availability",
    "book_appointment",
    "reschedule",
    "cancel",
    "take_message",
    "transfer_to_human",
)

_SERVICE_NAME = {
    "type": "string",
    "description": (
        "The exact name of one of the services this business offers, as listed "
        "in your instructions."
    ),
}
_PHONE = {
    "type": "string",
    "description": "The caller's phone number, as they gave it.",
}
_APPOINTMENT_ID = {
    "type": "string",
    "description": (
        "The appointment identifier returned by an earlier book_appointment "
        "result in this conversation."
    ),
}

_TOOLS: dict[str, tuple[str, dict[str, Any]]] = {
    "check_availability": (
        "List the start times a service can actually be booked at on one day. "
        "This is the only source of availability: call it before you offer a "
        "caller any time, and offer only the times it returns. An empty list "
        "means nothing is free that day.",
        {
            "service_name": _SERVICE_NAME,
            "day": {
                "type": "string",
                "description": "The day to look at, as an ISO date: 2026-03-02.",
            },
        },
    ),
    "book_appointment": (
        "Book an appointment. Only a start time that check_availability "
        "returned in this conversation can be booked. The booking has not "
        "happened unless this returns success — it can fail because someone "
        "else took the slot a moment earlier.",
        {
            "service_name": _SERVICE_NAME,
            "starts_at": {
                "type": "string",
                "description": (
                    "The start time, copied exactly from a check_availability "
                    "result: an ISO-8601 timestamp such as "
                    "2026-03-02T10:00:00+00:00."
                ),
            },
            "customer_name": {
                "type": "string",
                "description": "The caller's full name, as they gave it.",
            },
            "phone": _PHONE,
        },
    ),
    "reschedule": (
        "Move an existing appointment to a new start time. It stays the same "
        "appointment. The move has not happened unless this returns success.",
        {
            "appointment_id": _APPOINTMENT_ID,
            "new_starts_at": {
                "type": "string",
                "description": (
                    "The new start time as an ISO-8601 timestamp, taken from a "
                    "check_availability result."
                ),
            },
        },
    ),
    "cancel": (
        "Cancel an existing appointment. The cancellation has not happened "
        "unless this returns success.",
        {"appointment_id": _APPOINTMENT_ID},
    ),
    "take_message": (
        "Record a message for a human to deal with later. Use this when the "
        "caller wants something the calendar cannot do and does not want to be "
        "put through. All three details are required.",
        {
            "caller_name": {
                "type": "string",
                "description": "The caller's name, as they gave it.",
            },
            "phone": _PHONE,
            "message": {
                "type": "string",
                "description": "What the caller wants passed on, in their words.",
            },
        },
    ),
    "transfer_to_human": (
        "Hand the call to a person. Use this when the caller asks for someone, "
        "asks for medical or legal advice, is complaining or wants a refund, or "
        "has asked the same thing twice without getting anywhere.",
        {
            "reason": {
                "type": "string",
                "description": (
                    "Why this call needs a person, in one short sentence."
                ),
            }
        },
    ),
}


def _expected_arguments(name: str) -> set[str]:
    """The keyword-only arguments the registered tool actually takes."""
    parameters = inspect.signature(TOOLS[name]).parameters
    return {
        parameter.name
        for parameter in parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }


def tool_definitions() -> list[ToolDefinition]:
    """Every registered tool, described for the model.

    Built from `TOOLS` rather than from the table above, so a tool that gains
    an argument or joins the registry without a schema fails here instead of
    reaching a model as a silently wrong description.
    """
    missing = set(TOOLS) - set(_TOOLS)
    if missing:
        raise RuntimeError(
            f"No model-facing schema for registered tool(s): "
            f"{', '.join(sorted(missing))}."
        )
    unregistered = set(_TOOLS) - set(TOOLS)
    if unregistered:
        raise RuntimeError(
            f"Schema for tool(s) that are not registered: "
            f"{', '.join(sorted(unregistered))}."
        )

    definitions = []
    for name in TOOL_ORDER:
        description, properties = _TOOLS[name]
        if set(properties) != _expected_arguments(name):
            raise RuntimeError(
                f"The schema for {name!r} does not match its signature: "
                f"{sorted(properties)} against {sorted(_expected_arguments(name))}."
            )
        definitions.append(
            ToolDefinition(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": properties,
                    # Strict tool use requires both: every property listed as
                    # required, and nothing else accepted. None of the six
                    # tools has an optional argument, so nothing is lost.
                    "required": list(properties),
                    "additionalProperties": False,
                },
            )
        )
    return definitions
