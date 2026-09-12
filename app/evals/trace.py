"""What happened on one evaluated call, in memory and nowhere else.

The system under test writes its ordinary rows — `calls`, `turns`,
`tool_calls`, `appointments`, `call_costs` — because that is its job, and the
evaluation database exists so it can. The **trace** is different: it is the
evaluator's own reading of the call, it lives for the length of one run, and
it is never persisted. No evaluation table exists and none is proposed.

`ToolCallRecord` is reused verbatim rather than copied into a shape of the
evaluator's own. A second representation would eventually disagree with the
first, and the disagreement would be invisible.

The **offered-slot ledger** is the important part. `ToolExecutor` keeps one
privately to refuse unverified bookings; this reconstructs the same thing from
the trace, keyed identically, and keeps it per turn — because a time offered
to the caller on turn three does not excuse a claim made on turn one.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from app.dialogue import ToolCallRecord
from app.evals.scenario import AppointmentSpec, ScenarioError, slot_key

# How a whole call went wrong before any expectation could be judged.
PROVIDER_FAILURE = "PROVIDER_FAILURE"
SCRIPT_EXHAUSTED = "SCRIPT_EXHAUSTED"


@dataclass
class TurnTrace:
    """One caller turn and everything the system did about it."""

    index: int
    caller_text: str
    reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    escalated: bool = False
    booked_appointment_id: str | None = None
    failed: bool = False
    model_name: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    llm_latency_ms: int = 0
    # Realtime only. `None` on a text turn, which has no generation to be
    # stale and nothing to interrupt.
    generation: int | None = None
    interrupted: bool = False

    @property
    def offers(self) -> set[tuple[str, datetime]]:
        """Every (service, instant) a successful availability check returned.

        A slot that cannot be read is not remembered, which fails closed: the
        evaluator treats it as never offered rather than waving it through on
        a value nobody understood. `ToolExecutor` does the same.
        """
        found: set[tuple[str, datetime]] = set()
        for record in self.tool_calls:
            if record.tool_name != "check_availability" or not record.success:
                continue
            service = record.data.get("service")
            for slot in record.data.get("slots", []) or []:
                try:
                    found.add(slot_key(service, slot["starts_at"]))
                except (ScenarioError, KeyError, TypeError, ValueError):
                    continue
        return found


@dataclass
class CallTrace:
    """One scenario's call, as the evaluator saw it."""

    scenario: str
    kind: str
    call_id: uuid.UUID | None = None
    turns: list[TurnTrace] = field(default_factory=list)
    final_appointments: list[AppointmentSpec] = field(default_factory=list)
    # Which identifier each scenario `ref` became, so an expectation written
    # as `{{appointment:existing}}` can be compared with what was really
    # called. The script is substituted before the call; this is how the
    # ground truth is substituted after it.
    appointment_refs: dict[str, str] = field(default_factory=dict)
    total_cost_usd: Decimal | None = None
    component_costs: dict[str, Decimal | None] = field(default_factory=dict)
    model_calls: int = 0
    # Set when the call could not be completed at all. `error_category` says
    # whose fault that was — the dataset's or the system's.
    error: str | None = None
    error_category: str | None = None
    # Realtime only.
    generation: int | None = None
    sink_chunks: int = 0
    sink_clears: int = 0

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        """Every tool call of the whole call, in the order they happened."""
        return [record for turn in self.turns for record in turn.tool_calls]

    @property
    def escalated(self) -> bool:
        """Did a `transfer_to_human` actually succeed on any turn?"""
        return any(turn.escalated for turn in self.turns)

    @property
    def offered(self) -> set[tuple[str, datetime]]:
        """The whole call's ledger."""
        return self.offered_through(len(self.turns))

    def offered_through(self, turns: int) -> set[tuple[str, datetime]]:
        """The ledger as it stood at the end of turn `turns - 1`.

        A prefix, because the question a claim has to answer is not "was this
        ever offered?" but "had it been offered *yet*?".
        """
        found: set[tuple[str, datetime]] = set()
        for turn in self.turns[:turns]:
            found |= turn.offers
        return found

    @property
    def booked_turn(self) -> int | None:
        """The 1-based turn on which a booking succeeded, if one did."""
        for turn in self.turns:
            if turn.booked_appointment_id is not None:
                return turn.index + 1
        return None

    @property
    def booked_appointment_id(self) -> str | None:
        for turn in self.turns:
            if turn.booked_appointment_id is not None:
                return turn.booked_appointment_id
        return None

    @property
    def failed_run(self) -> bool:
        """Did the call fall over before it could be judged?"""
        return self.error_category is not None


__all__ = [
    "PROVIDER_FAILURE",
    "SCRIPT_EXHAUSTED",
    "CallTrace",
    "TurnTrace",
]
