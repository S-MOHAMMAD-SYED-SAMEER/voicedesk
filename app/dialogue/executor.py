"""Running the tool the model asked for — and refusing the ones it may not.

Everything callable goes through `app.tools.get_tool`. This module adds no
tool of its own, reaches no database, and knows nothing about calendars. What
it does add is the one guard the prompt cannot enforce:

    `book_appointment` may only run for a start time that a *successful*
    `check_availability` returned earlier in this conversation.

The specification requires a hallucinated-availability rate of zero. A prompt
can ask for that; only code can guarantee it. The ledger below is that code —
an in-memory record of what the calendar has actually offered, which the model
cannot write to except by asking the calendar. It is a floor, not a ceiling:
`CalendarService` and the database's exclusion constraint remain the authority
on whether a verified slot is still free by the time anyone writes to it.

What the guard does not cover, deliberately: whether the caller agreed. That
is a conversation, not a fact, and it stays a behavioural rule.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.providers.llm import ToolUse
from app.tools import ToolContext, ToolResult, UnknownTool, get_tool
from app.tools.base import ToolArgumentError, parse_instant, require_text


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool invocation, in the shape `tool_calls` stores.

    `data` is kept whole here even when the call failed — the model is told
    about `services_offered` and `slot_taken` so it can recover. What reaches
    the database is narrower; see `Conversation._persist`.
    """

    tool_name: str
    arguments: dict[str, Any]
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: int | None = None


UNVERIFIED_SLOT = (
    "That start time was not offered by check_availability in this "
    "conversation, so it has not been verified and cannot be booked. Call "
    "check_availability for the service and day first, then book one of the "
    "times it returns."
)


class ToolExecutor:
    """Dispatches model tool calls, one conversation's worth."""

    def __init__(self, context: ToolContext) -> None:
        self._context = context
        # (service name, instant) pairs the calendar has actually offered.
        self._offered: set[tuple[str, datetime]] = set()

    # --- the ledger --------------------------------------------------------

    @property
    def offered_slots(self) -> set[tuple[str, datetime]]:
        return set(self._offered)

    def _key(self, service_name: Any, starts_at: Any) -> tuple[str, datetime]:
        """Normalise a service and a time to something comparable.

        Instants, not strings: the model may write an offered time back with a
        different but equivalent offset, and that is the same moment.
        """
        name = require_text(service_name, "service_name").casefold()
        moment = parse_instant(starts_at, self._context.timezone())
        return name, moment.astimezone(UTC)

    def _record_offer(self, result: ToolResult) -> None:
        """Remember what a successful availability check offered."""
        service = result.data.get("service")
        for slot in result.data.get("slots", []):
            try:
                self._offered.add(self._key(service, slot["starts_at"]))
            except (ToolArgumentError, KeyError, TypeError):
                # A slot we cannot parse is simply not remembered, which fails
                # closed: the model is refused that booking rather than waved
                # through on a value nobody understood.
                continue

    def _verified(self, arguments: dict[str, Any]) -> bool:
        """Has this exact service and start time been offered?

        Arguments that cannot be normalised are not treated as a guard
        failure: the tool itself produces a much better message for them, so
        the call is allowed through to be refused there.
        """
        try:
            key = self._key(arguments.get("service_name"), arguments.get("starts_at"))
        except (ToolArgumentError, TypeError):
            return True
        return key in self._offered

    # --- dispatch ----------------------------------------------------------

    def execute(self, tool_use: ToolUse) -> tuple[dict[str, Any], ToolCallRecord]:
        """Run one tool call, returning what to send back and what to store."""
        arguments = tool_use.arguments if isinstance(tool_use.arguments, dict) else {}

        if not isinstance(tool_use.arguments, dict):
            return self._finish(
                tool_use,
                ToolResult.failed(
                    f"Arguments for {tool_use.name!r} must be an object."
                ),
                arguments,
                latency_ms=0,
            )

        try:
            tool = get_tool(tool_use.name)
        except UnknownTool as exc:
            # Recorded, not swallowed: the wrong-tool-call rate is a metric.
            return self._finish(
                tool_use, ToolResult.failed(str(exc)), arguments, latency_ms=0
            )

        if tool_use.name == "book_appointment" and not self._verified(arguments):
            return self._finish(
                tool_use,
                ToolResult.failed(UNVERIFIED_SLOT, unverified_slot=True),
                arguments,
                latency_ms=0,
            )

        started = time.perf_counter()
        try:
            result = tool(self._context, **arguments)
        except TypeError as exc:
            # The model sent an argument the tool does not take, or left one
            # out. Strict schemas should prevent it; a returned failure means
            # the turn survives if one ever gets through.
            result = ToolResult.failed(
                f"{tool_use.name} could not be called with those arguments: {exc}"
            )
        latency_ms = int((time.perf_counter() - started) * 1000)

        if tool_use.name == "check_availability" and result.success:
            self._record_offer(result)

        return self._finish(tool_use, result, arguments, latency_ms=latency_ms)

    def _finish(
        self,
        tool_use: ToolUse,
        result: ToolResult,
        arguments: dict[str, Any],
        latency_ms: int | None,
    ) -> tuple[dict[str, Any], ToolCallRecord]:
        block = {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps(result.as_dict()),
            "is_error": not result.success,
        }
        record = ToolCallRecord(
            tool_name=tool_use.name,
            arguments=arguments,
            success=result.success,
            data=result.data,
            error=result.error,
            latency_ms=latency_ms,
        )
        return block, record
