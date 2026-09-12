"""Turning a trace and its ground truth into findings, deterministically.

Every finding names a category, what was expected, what was observed, and the
evidence — so a report can say *FAIL because X happened but Y was expected*
without anyone having to read prose and decide.

No model is asked anything here. There is no judge, no scoring heuristic and
no natural-language understanding: every check is a comparison of values the
system actually produced against values the scenario actually declared.

Two categories are deliberately not blocking:

* `UNVERIFIED_RESCHEDULE` — a reschedule to a time the calendar never offered.
  The executor's guard covers `book_appointment` and not `reschedule`, which
  is frozen milestone-3/4 behaviour. The suite measures it and reports it; it
  does not fail a scenario for behaving exactly as the system is built to.
* `SCRIPT_EXHAUSTED` — a scenario that ran out of script. That is a fault in
  the dataset, and attributing it to the receptionist would be a lie about
  what was measured.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import time
from typing import Any

from app.dialogue import ToolCallRecord
from app.evals.scenario import (
    ExpectedTool,
    Expectation,
    Scenario,
    arguments_match,
    normalise_argument,
    placeholder_ref,
    slot_key,
    timezone,
)
from app.evals.trace import SCRIPT_EXHAUSTED, CallTrace

# --- the taxonomy ----------------------------------------------------------

WRONG_TOOL = "WRONG_TOOL"
MISSING_TOOL = "MISSING_TOOL"
INVALID_TOOL_ARGUMENT = "INVALID_TOOL_ARGUMENT"
HALLUCINATED_AVAILABILITY = "HALLUCINATED_AVAILABILITY"
UNVERIFIED_RESCHEDULE = "UNVERIFIED_RESCHEDULE"
UNDECLARED_CLAIM = "UNDECLARED_CLAIM"
MISSED_ESCALATION = "MISSED_ESCALATION"
UNNECESSARY_ESCALATION = "UNNECESSARY_ESCALATION"
BOOKING_FAILURE = "BOOKING_FAILURE"
RESCHEDULE_FAILURE = "RESCHEDULE_FAILURE"
CANCELLATION_FAILURE = "CANCELLATION_FAILURE"
DUPLICATE_ACTION = "DUPLICATE_ACTION"
UNEXPECTED_ACTION = "UNEXPECTED_ACTION"
TURN_LIMIT = "TURN_LIMIT"
PROVIDER_FAILURE = "PROVIDER_FAILURE"

CATEGORIES = (
    WRONG_TOOL,
    MISSING_TOOL,
    INVALID_TOOL_ARGUMENT,
    HALLUCINATED_AVAILABILITY,
    UNVERIFIED_RESCHEDULE,
    UNDECLARED_CLAIM,
    MISSED_ESCALATION,
    UNNECESSARY_ESCALATION,
    BOOKING_FAILURE,
    RESCHEDULE_FAILURE,
    CANCELLATION_FAILURE,
    DUPLICATE_ACTION,
    UNEXPECTED_ACTION,
    TURN_LIMIT,
    PROVIDER_FAILURE,
    SCRIPT_EXHAUSTED,
)

# Measured, reported, and never counted against task success. See the module
# docstring for why each is here.
NON_BLOCKING = frozenset({UNVERIFIED_RESCHEDULE})

# Blocking, but blamed on the dataset rather than on VoiceDesk.
AUTHORING = frozenset({SCRIPT_EXHAUSTED, UNDECLARED_CLAIM})

# The task each failed outcome belongs to.
TASK_FAILURE = {
    "book": BOOKING_FAILURE,
    "reschedule": RESCHEDULE_FAILURE,
    "cancel": CANCELLATION_FAILURE,
}


@dataclass(frozen=True)
class Finding:
    """One thing that was not as the scenario said it would be."""

    category: str
    expected: str
    observed: str
    evidence: str = ""

    @property
    def blocking(self) -> bool:
        return self.category not in NON_BLOCKING

    @property
    def authoring(self) -> bool:
        """Is this the dataset's fault rather than the system's?"""
        return self.category in AUTHORING


@dataclass
class ToolTally:
    """How the observed tool calls lined up with the expected ones."""

    correct: int = 0
    wrong: int = 0
    missing: int = 0
    duplicate: int = 0
    invalid_argument: int = 0

    @property
    def total(self) -> int:
        return (
            self.correct
            + self.wrong
            + self.missing
            + self.duplicate
            + self.invalid_argument
        )

    def __iadd__(self, other: "ToolTally") -> "ToolTally":
        self.correct += other.correct
        self.wrong += other.wrong
        self.missing += other.missing
        self.duplicate += other.duplicate
        self.invalid_argument += other.invalid_argument
        return self


@dataclass
class Assessment:
    """One scenario, judged."""

    scenario: str
    findings: list[Finding] = field(default_factory=list)
    tools: ToolTally = field(default_factory=ToolTally)

    @property
    def blocking(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.blocking]

    @property
    def passed(self) -> bool:
        return not self.blocking

    @property
    def observations(self) -> list[Finding]:
        return [finding for finding in self.findings if not finding.blocking]


def assess(scenario: Scenario, trace: CallTrace) -> Assessment:
    """Every check, against one call."""
    assessment = Assessment(scenario=scenario.name)

    if trace.failed_run:
        assessment.findings.append(
            Finding(
                category=trace.error_category or PROVIDER_FAILURE,
                expected="the call to run to completion",
                observed="it did not",
                evidence=trace.error or "",
            )
        )
        # Nothing further can be judged honestly about a call that never ran.
        return assessment

    expect = scenario.expect

    assessment.tools = _tools(expect, trace, assessment.findings)
    _escalation(expect, trace, assessment.findings)
    _availability(expect, trace, assessment.findings)
    _undeclared_claims(expect, trace, assessment.findings)
    _unverified_reschedules(trace, assessment.findings)
    _final_state(expect, trace, assessment.findings)
    _turn_limit(scenario, trace, assessment.findings)
    _provider(expect, trace, assessment.findings)
    _model_calls(expect, trace, assessment.findings)
    _realtime(expect, trace, assessment.findings)

    return assessment


# --- tools -----------------------------------------------------------------


def _tools(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> ToolTally:
    """Match observed calls to expected ones, and account for the rest."""
    tally = ToolTally()
    remaining = [_resolved(expected, trace) for expected in expect.expected_tools]
    matched: list[ToolCallRecord] = []
    unmatched: list[ToolCallRecord] = []

    for record in trace.tool_calls:
        found = next(
            (
                candidate
                for candidate in remaining
                if candidate.name == record.tool_name
                and candidate.succeeds == record.success
                and arguments_match(candidate.arguments, record.arguments)
            ),
            None,
        )
        if found is None:
            unmatched.append(record)
            continue
        remaining.remove(found)
        matched.append(record)
        tally.correct += 1

    for expected in remaining:
        tally.missing += 1
        findings.append(
            Finding(
                category=MISSING_TOOL,
                expected=_describe_expected(expected),
                observed="it was never called",
                evidence=_observed_tools(trace),
            )
        )

    expected_names = {expected.name for expected in expect.expected_tools}
    resolved = tuple(
        _resolved(expected, trace) for expected in expect.expected_tools
    )
    for record in unmatched:
        tally_category = _classify(
            record, expect, expected_names, matched, resolved
        )
        setattr(tally, tally_category[1], getattr(tally, tally_category[1]) + 1)
        findings.append(tally_category[0])

    return tally


def _resolved(expected: ExpectedTool, trace: CallTrace) -> ExpectedTool:
    """One expectation with its `{{appointment:ref}}` placeholders filled in.

    The script is substituted before the call so the tools receive real
    identifiers; the ground truth has to be substituted after it, against the
    same mapping, or every seeded-appointment expectation would compare a
    placeholder against a UUID and never match.
    """
    arguments = dict(expected.arguments)
    changed = False
    for key, value in arguments.items():
        ref = placeholder_ref(value)
        if ref is not None and ref in trace.appointment_refs:
            arguments[key] = trace.appointment_refs[ref]
            changed = True
    if not changed:
        return expected
    return replace(expected, arguments=arguments)


def _classify(
    record: ToolCallRecord,
    expect: Expectation,
    expected_names: set[str],
    matched: Sequence[ToolCallRecord],
    resolved: Sequence[ExpectedTool],
) -> tuple[Finding, str]:
    """Why this observed call did not match anything expected."""
    if record.tool_name in expect.forbidden_tools:
        return (
            Finding(
                category=WRONG_TOOL,
                expected=f"{record.tool_name} never to be called",
                observed=f"{record.tool_name} was called",
                evidence=_describe_record(record),
            ),
            "wrong",
        )

    if any(
        earlier.tool_name == record.tool_name
        and record.success
        and earlier.success
        and _same_arguments(earlier.arguments, record.arguments)
        for earlier in matched
    ):
        return (
            Finding(
                category=DUPLICATE_ACTION,
                expected=f"one successful {record.tool_name}",
                observed=f"{record.tool_name} succeeded again with the same arguments",
                evidence=_describe_record(record),
            ),
            "duplicate",
        )

    if record.tool_name in expected_names:
        # The right tool, but not as any expectation described it — including
        # an expected-to-fail call whose arguments did not match.
        return (
            Finding(
                category=INVALID_TOOL_ARGUMENT,
                expected=_expected_arguments_for(resolved, record.tool_name),
                observed=_describe_record(record),
                evidence=f"{record.tool_name} was called, but not as expected",
            ),
            "invalid_argument",
        )

    if not record.success and record.tool_name in expect.allow_tool_failures:
        return (
            Finding(
                category=UNEXPECTED_ACTION,
                expected=f"{record.tool_name} to fail as declared",
                observed="it failed, but no expectation matched it",
                evidence=_describe_record(record),
            ),
            "wrong",
        )

    return (
        Finding(
            category=UNEXPECTED_ACTION,
            expected="no tool beyond those declared",
            observed=f"{record.tool_name} was called",
            evidence=_describe_record(record),
        ),
        "wrong",
    )


# --- escalation ------------------------------------------------------------


def _escalation(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> None:
    """`DialogueResult.escalated` is true only when a transfer succeeded."""
    if expect.should_escalate and not trace.escalated:
        findings.append(
            Finding(
                category=MISSED_ESCALATION,
                expected="the call to be escalated to a human",
                observed="it was not",
                evidence=_observed_tools(trace) or "no tools were called",
            )
        )
    if trace.escalated and not expect.should_escalate:
        findings.append(
            Finding(
                category=UNNECESSARY_ESCALATION,
                expected="the call to be handled without a human",
                observed="it was escalated",
                evidence=_escalation_reason(trace),
            )
        )


# --- hallucinated availability ---------------------------------------------


def _availability(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> None:
    """Layers one and two: structural bookings, then declared claims."""
    # L1. The executor's guard should make this impossible. The suite asserts
    # that it stays impossible rather than assuming it.
    ledger = trace.offered
    for record in trace.tool_calls:
        if record.tool_name != "book_appointment" or not record.success:
            continue
        try:
            key = slot_key(
                record.arguments.get("service_name"),
                record.arguments.get("starts_at"),
            )
        except Exception:  # noqa: BLE001 - an unreadable booking is a finding
            key = None
        if key is None or key not in ledger:
            findings.append(
                Finding(
                    category=HALLUCINATED_AVAILABILITY,
                    expected="a booking only at a time check_availability offered",
                    observed="a booking succeeded at a time never offered",
                    evidence=_describe_record(record),
                )
            )

    # L2. What the scripted replies told the caller was free, checked against
    # the ledger as it stood when they said it.
    for claim in expect.availability_claims:
        known = trace.offered_through(claim.turn + 1)
        try:
            key = slot_key(claim.service, claim.starts_at)
        except Exception:  # noqa: BLE001 - validated at load, defensive here
            key = None
        if key is None or key not in known:
            findings.append(
                Finding(
                    category=HALLUCINATED_AVAILABILITY,
                    expected=(
                        f"{claim.service} at {claim.starts_at} to have been "
                        f"offered by turn {claim.turn + 1}"
                    ),
                    observed="no successful availability check had offered it",
                    evidence=_describe_ledger(known),
                )
            )


# --- the undeclared-claim detector -----------------------------------------

# Deliberately narrow. It finds unambiguous clock times only — `10:00`,
# `10am`, `3:30 pm` — and never words like "ten" or "half nine". Its job is to
# catch a scenario author who put a time in a reply and forgot to declare it,
# not to decide whether a sentence means something. A detector that guessed
# would produce findings nobody could check.
CLOCK = re.compile(
    r"(?<![\d:])(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)\s*(?P<suffix>am|pm)?"
    r"(?![\d:])",
    re.IGNORECASE,
)
HOUR_ONLY = re.compile(
    r"(?<![\d:.])(?P<hour>1[0-2]|[1-9])\s*(?P<suffix>am|pm)(?![\d:])",
    re.IGNORECASE,
)


def clock_times(text: str) -> set[time]:
    """Every unambiguous clock time in a reply, as times of day."""
    found: set[time] = set()
    for match in CLOCK.finditer(text):
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))
        hour = _apply_suffix(hour, match.group("suffix"))
        if hour is None or hour > 23:
            continue
        found.add(time(hour, minute))
    for match in HOUR_ONLY.finditer(text):
        hour = _apply_suffix(int(match.group("hour")), match.group("suffix"))
        if hour is None:
            continue
        found.add(time(hour, 0))
    return found


def _minutes(moment: time) -> str:
    return moment.isoformat(timespec="minutes")


def _apply_suffix(hour: int, suffix: str | None) -> int | None:
    if suffix is None:
        return hour if hour <= 23 else None
    lowered = suffix.lower()
    if not 1 <= hour <= 12:
        return None
    if lowered == "am":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def _undeclared_claims(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> None:
    """Layer three: a time in a reply that nothing accounts for.

    Accounted for means: offered by the calendar by that turn, or declared by
    the scenario as a claim on that turn or an earlier one. Anything else is a
    scenario that says something its ground truth does not mention, which
    makes its hallucination result meaningless — so it is reported.
    """
    zone = timezone()
    for turn in trace.turns:
        spoken = clock_times(turn.reply)
        if not spoken:
            continue

        allowed = {
            instant.astimezone(zone).timetz().replace(tzinfo=None)
            for _, instant in trace.offered_through(turn.index + 1)
        }
        allowed |= {
            slot_key(claim.service, claim.starts_at)[1]
            .astimezone(zone)
            .timetz()
            .replace(tzinfo=None)
            for claim in expect.availability_claims
            if claim.turn <= turn.index
        }

        for moment in sorted(spoken - allowed):
            findings.append(
                Finding(
                    category=UNDECLARED_CLAIM,
                    expected="every time named in a reply to be offered or declared",
                    observed=f"the reply names {moment.isoformat(timespec='minutes')}",
                    evidence=(
                        f"turn {turn.index + 1}: {turn.reply!r}; "
                        f"accounted for: "
                        f"{sorted(_minutes(item) for item in allowed)}"
                    ),
                )
            )


# --- the reschedule observation --------------------------------------------


def _unverified_reschedules(trace: CallTrace, findings: list[Finding]) -> None:
    """A reschedule to a time the calendar never offered.

    Measured, not blamed. `ToolExecutor` guards `book_appointment` and not
    `reschedule`; that is the frozen behaviour of milestones 3 and 4, and this
    suite reports it rather than failing a scenario for it.
    """
    ledger = {instant for _, instant in trace.offered}
    for record in trace.tool_calls:
        if record.tool_name != "reschedule" or not record.success:
            continue
        try:
            _, moment = slot_key("x", record.arguments.get("new_starts_at"))
        except Exception:  # noqa: BLE001 - unreadable means unverified
            moment = None
        if moment is None or moment not in ledger:
            findings.append(
                Finding(
                    category=UNVERIFIED_RESCHEDULE,
                    expected="a reschedule only to a time the calendar offered",
                    observed="the new time was never offered in this call",
                    evidence=_describe_record(record),
                )
            )


# --- final state -----------------------------------------------------------


def _final_state(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> None:
    """What `appointments` holds, compared exactly against the ground truth."""
    observed = sorted(_normalise_appointments(trace.final_appointments))
    wanted = sorted(_normalise_appointments(expect.final_appointments))
    if observed == wanted:
        return

    category = TASK_FAILURE.get(expect.task, UNEXPECTED_ACTION)
    findings.append(
        Finding(
            category=category,
            expected=_describe_appointments(wanted),
            observed=_describe_appointments(observed),
            evidence="the appointments table after the call",
        )
    )


def _normalise_appointments(specs: Sequence[Any]) -> list[tuple[str, str, Any, str]]:
    return [
        (
            spec.service_name.casefold(),
            spec.customer_name.casefold(),
            normalise_argument(spec.starts_at),
            spec.status,
        )
        for spec in specs
    ]


# --- the rest --------------------------------------------------------------


def _turn_limit(scenario: Scenario, trace: CallTrace, findings: list[Finding]) -> None:
    if len(trace.turns) > scenario.max_turns:
        findings.append(
            Finding(
                category=TURN_LIMIT,
                expected=f"at most {scenario.max_turns} turns",
                observed=f"{len(trace.turns)} turns",
                evidence="",
            )
        )


def _provider(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> None:
    """A turn the dialogue layer itself reported as failed."""
    del expect
    for turn in trace.turns:
        if not turn.failed:
            continue
        findings.append(
            Finding(
                category=PROVIDER_FAILURE,
                expected="every turn to complete",
                observed=f"turn {turn.index + 1} failed",
                evidence=turn.reply,
            )
        )


def _model_calls(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> None:
    """How many times the model was asked, when the scenario says."""
    if expect.expected_model_calls is None:
        return
    if trace.model_calls == expect.expected_model_calls:
        return
    findings.append(
        Finding(
            category=DUPLICATE_ACTION
            if trace.model_calls > expect.expected_model_calls
            else UNEXPECTED_ACTION,
            expected=f"{expect.expected_model_calls} model requests",
            observed=f"{trace.model_calls} model requests",
            evidence="a repeated request would mean work was done twice",
        )
    )


def _realtime(
    expect: Expectation, trace: CallTrace, findings: list[Finding]
) -> None:
    """What a streaming call must show, when the scenario says it should.

    These are assertions about milestone 7's generation model, made from
    outside it: the session's own counters and the sink's own record. Nothing
    in M7 was changed to expose them.
    """
    wanted = expect.realtime
    if wanted is None:
        return

    if wanted.turns is not None and len(trace.turns) != wanted.turns:
        findings.append(
            Finding(
                category=UNEXPECTED_ACTION,
                expected=f"{wanted.turns} turns",
                observed=f"{len(trace.turns)} turns",
                evidence="silence must not produce a turn at all",
            )
        )

    interrupted = any(turn.interrupted for turn in trace.turns)
    if interrupted != wanted.interrupted:
        findings.append(
            Finding(
                category=UNEXPECTED_ACTION,
                expected=(
                    "a turn marked interrupted"
                    if wanted.interrupted
                    else "no turn to be interrupted"
                ),
                observed=f"interrupted={interrupted}",
                evidence="RealtimeTurn.interrupted",
            )
        )

    if wanted.generation_advances:
        generation = trace.generation or 0
        if generation <= len(trace.turns):
            findings.append(
                Finding(
                    category=UNEXPECTED_ACTION,
                    expected="the generation to advance past the turn that was cut off",
                    observed=f"generation {generation} after {len(trace.turns)} turns",
                    evidence="a stale reply is only discardable once it is stale",
                )
            )

    if wanted.audio_discarded and not (
        trace.sink_clears >= 1 and trace.sink_chunks == 0
    ):
        findings.append(
            Finding(
                category=UNEXPECTED_ACTION,
                expected="queued audio discarded, not merely stopped",
                observed=(
                    f"{trace.sink_clears} clears, {trace.sink_chunks} chunks left"
                ),
                evidence="the caller must not hear the rest of what they cut off",
            )
        )


# --- describing things -----------------------------------------------------


def _same_arguments(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        normalise_argument(left[key]) == normalise_argument(right[key])
        for key in left
    )


def _describe_record(record: ToolCallRecord) -> str:
    outcome = "succeeded" if record.success else f"failed: {record.error}"
    return f"{record.tool_name}({_arguments(record.arguments)}) {outcome}"


def _describe_expected(expected: Any) -> str:
    outcome = "succeeding" if expected.succeeds else "failing"
    return f"{expected.name}({_arguments(dict(expected.arguments))}) {outcome}"


def _expected_arguments_for(resolved: Sequence[Any], name: str) -> str:
    return " or ".join(
        _describe_expected(expected)
        for expected in resolved
        if expected.name == name
    )


def _arguments(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in sorted(arguments.items()))


def _observed_tools(trace: CallTrace) -> str:
    return " → ".join(
        f"{record.tool_name}{'' if record.success else '(failed)'}"
        for record in trace.tool_calls
    )


def _escalation_reason(trace: CallTrace) -> str:
    for record in trace.tool_calls:
        if record.tool_name == "transfer_to_human" and record.success:
            return str(record.data.get("reason", ""))
    return ""


def _describe_ledger(ledger: set[tuple[str, Any]]) -> str:
    if not ledger:
        return "nothing had been offered"
    return "offered: " + ", ".join(
        f"{service} at {instant.isoformat()}" for service, instant in sorted(ledger)
    )


def _describe_appointments(rows: Sequence[tuple[str, str, Any, str]]) -> str:
    if not rows:
        return "no appointments"
    return "; ".join(
        f"{service} for {customer} at {instant} ({status})"
        for service, customer, instant, status in rows
    )


__all__ = [
    "AUTHORING",
    "BOOKING_FAILURE",
    "CANCELLATION_FAILURE",
    "CATEGORIES",
    "DUPLICATE_ACTION",
    "HALLUCINATED_AVAILABILITY",
    "INVALID_TOOL_ARGUMENT",
    "MISSED_ESCALATION",
    "MISSING_TOOL",
    "NON_BLOCKING",
    "PROVIDER_FAILURE",
    "RESCHEDULE_FAILURE",
    "TURN_LIMIT",
    "UNDECLARED_CLAIM",
    "UNEXPECTED_ACTION",
    "UNNECESSARY_ESCALATION",
    "UNVERIFIED_RESCHEDULE",
    "WRONG_TOOL",
    "Assessment",
    "Finding",
    "ToolTally",
    "assess",
    "clock_times",
]
