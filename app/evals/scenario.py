"""What a scenario is: a world, a script, and structured ground truth.

A scenario is **data**, not a test function. It says what the business looks
like, what the caller says, what the model says back, and — separately, in
structured form — what must be true afterwards. Nothing here is prose that a
human has to interpret: every expectation is a value the evaluator can compare.

Two things are worth knowing before reading further.

**Times are compared as instants, never as strings.** The same moment can be
written several ways, and a scenario that expected one spelling would fail on
another. Normalisation goes through `app.tools.base.parse_instant` — the same
function the tool layer uses — so the evaluator and the system under test
agree by construction rather than by coincidence.

**Seeded appointments are referred to by name.** A scenario cannot know the
identifier of a row that does not exist yet, so a script writes
`{{appointment:existing}}` and the runner substitutes the real one once the
world has been built. A placeholder that names nothing is a dataset error, not
a silent `None`.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from app.providers.llm import ModelResponse
from app.tools import TOOLS
from app.tools.base import ToolArgumentError, parse_instant

# Every scenario runs in this timezone whatever the deployment uses, because
# the dataset's fixed dates have to mean the same thing everywhere.
EVAL_TIMEZONE = "UTC"

KINDS = ("text", "realtime")
TASKS = ("book", "reschedule", "cancel", "message", "escalate", "decline")

PLACEHOLDER = re.compile(r"^\{\{appointment:([a-z0-9_]+)\}\}$")


class ScenarioError(ValueError):
    """A scenario that cannot be run as written. A dataset fault, not a bug."""


# --- the world -------------------------------------------------------------


@dataclass(frozen=True)
class ServiceSpec:
    """One row of `services`."""

    name: str
    duration_minutes: int = 30
    staff_id: str = "sam"
    active: bool = True


@dataclass(frozen=True)
class HoursSpec:
    """One row of `business_hours`. Monday is 0."""

    weekday: int
    opens_at: time = time(9)
    closes_at: time = time(17)


@dataclass(frozen=True)
class SeedAppointment:
    """An appointment that already exists when the call starts.

    `ref` is how the script names it before it has an identifier.
    """

    ref: str
    service_name: str
    customer_name: str
    phone: str
    starts_at: str
    cancelled: bool = False


@dataclass(frozen=True)
class World:
    """Everything the business looks like before the phone rings."""

    services: tuple[ServiceSpec, ...]
    hours: tuple[HoursSpec, ...]
    appointments: tuple[SeedAppointment, ...] = ()
    from_number: str = "+447700900123"
    to_number: str = "+441234567890"

    @property
    def refs(self) -> frozenset[str]:
        return frozenset(seed.ref for seed in self.appointments)


def weekdays(
    opens_at: time = time(9), closes_at: time = time(17)
) -> tuple[HoursSpec, ...]:
    """Monday to Friday, the same hours each day."""
    return tuple(HoursSpec(day, opens_at, closes_at) for day in range(5))


# --- the script ------------------------------------------------------------


@dataclass(frozen=True)
class Beat:
    """One caller utterance, and every model response it provokes.

    A turn may go round the tool loop several times, so `responses` is a
    sequence: one per request the model would make for this utterance.
    """

    caller: str
    responses: tuple[ModelResponse, ...]


# --- ground truth ----------------------------------------------------------


@dataclass(frozen=True)
class ExpectedTool:
    """A tool call that must happen, and the arguments it must carry.

    `arguments` is a **subset**: a scenario asserts the fields it cares about
    and stays silent about the rest, so adding a field to a tool does not
    invalidate the dataset.
    """

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    succeeds: bool = True


@dataclass(frozen=True)
class Claim:
    """A time the scripted reply tells the caller is available.

    Declared per turn, because a claim made before the calendar was asked is
    exactly the failure this suite exists to catch — and checking against the
    ledger as it stood at the end of the call would miss it.
    """

    service: str
    starts_at: str
    turn: int


@dataclass(frozen=True)
class AppointmentSpec:
    """One row `appointments` must contain when the call is over."""

    service_name: str
    customer_name: str
    starts_at: str
    status: str = "booked"


@dataclass(frozen=True)
class RealtimeExpectation:
    """What a streaming call must show, beyond what any call must.

    Only the realtime scenarios carry one. `turns` of `None` means the count
    is not asserted; everything else defaults to "did not happen", which is
    what an ordinary uninterrupted call looks like.
    """

    turns: int | None = None
    interrupted: bool = False
    # The session bumps its generation once per turn, and once more when
    # somebody talks over a reply. So "more generations than turns" is
    # precisely "an interruption happened".
    generation_advances: bool = False
    # Queued audio was discarded rather than merely stopped: the caller does
    # not hear the rest of a sentence they interrupted.
    audio_discarded: bool = False


@dataclass(frozen=True)
class Expectation:
    """What must be true when the call ends. All of it, structurally."""

    task: str
    expected_tools: tuple[ExpectedTool, ...] = ()
    forbidden_tools: frozenset[str] = frozenset()
    should_escalate: bool = False
    availability_claims: tuple[Claim, ...] = ()
    final_appointments: tuple[AppointmentSpec, ...] = ()
    # Tools this scenario expects to fail. A failure anywhere else is a
    # finding: a system that silently tolerates broken tools is the thing the
    # specification is most afraid of.
    allow_tool_failures: frozenset[str] = frozenset()
    # Model calls expected across the whole call. `None` means "not asserted".
    # An interrupted turn that ran the model twice would show up here.
    expected_model_calls: int | None = None
    # Streaming scenarios only.
    realtime: RealtimeExpectation | None = None


# --- a scenario ------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One evaluated call."""

    name: str
    kind: str
    description: str
    world: World
    script: tuple[Beat, ...]
    expect: Expectation
    max_turns: int = 8
    # Realtime only: how many 20 ms frames of speech and of silence to feed,
    # and whether to interrupt the reply once it is playing.
    speech_frames: int = 10
    silence_frames: int = 40
    interrupt: bool = False

    @property
    def responses(self) -> tuple[ModelResponse, ...]:
        """Every scripted response, in the order the model will produce them."""
        return tuple(
            response for beat in self.script for response in beat.responses
        )


# --- normalisation ---------------------------------------------------------


def timezone() -> ZoneInfo:
    return ZoneInfo(EVAL_TIMEZONE)


def normalise_instant(value: Any) -> datetime:
    """One moment, in UTC, however it was written.

    The same call the tool layer makes, so a scenario and the system under
    test cannot disagree about what a timestamp means.
    """
    return parse_instant(value, timezone()).astimezone(UTC)


def slot_key(service: Any, starts_at: Any) -> tuple[str, datetime]:
    """The identity the offered-slot ledger is keyed by.

    Deliberately the same shape as `ToolExecutor._key`: a case-folded service
    name and an instant. If the two ever diverge the evaluator would be
    measuring something the executor is not enforcing.
    """
    name = str(service).strip().casefold()
    if not name:
        raise ScenarioError("A slot key needs a service name.")
    return name, normalise_instant(starts_at)


def normalise_argument(value: Any) -> Any:
    """A tool argument, in a form two spellings of it compare equal in."""
    if isinstance(value, datetime):
        return normalise_instant(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return normalise_instant(text)
        except (ToolArgumentError, ValueError):
            return text.casefold()
    return value


def arguments_match(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """Does `observed` carry everything `expected` asks for, equivalently?"""
    for key, wanted in expected.items():
        if key not in observed:
            return False
        if normalise_argument(wanted) != normalise_argument(observed[key]):
            return False
    return True


def placeholder_ref(value: Any) -> str | None:
    """The appointment this value stands in for, if it stands in for one."""
    if not isinstance(value, str):
        return None
    found = PLACEHOLDER.match(value.strip())
    return found.group(1) if found else None


# --- validation ------------------------------------------------------------


def validate(scenario: Scenario) -> None:
    """Refuse a scenario that cannot be scored, before anything runs.

    Every failure here is a dataset fault. Finding them at load time means a
    malformed expectation is never mistaken at run time for a system failure.
    """
    if not scenario.name:
        raise ScenarioError("A scenario needs a name.")
    if scenario.kind not in KINDS:
        raise ScenarioError(
            f"{scenario.name}: kind {scenario.kind!r} is not one of "
            f"{', '.join(KINDS)}."
        )
    if scenario.expect.task not in TASKS:
        raise ScenarioError(
            f"{scenario.name}: task {scenario.expect.task!r} is not one of "
            f"{', '.join(TASKS)}."
        )
    if scenario.max_turns <= 0:
        raise ScenarioError(f"{scenario.name}: max_turns must be positive.")

    _validate_world(scenario)
    _validate_tools(scenario)
    _validate_claims(scenario)
    _validate_appointments(scenario)
    _validate_placeholders(scenario)


def validate_all(scenarios: Sequence[Scenario], *, expected: int | None = None) -> None:
    """The whole dataset: every scenario valid, and no two the same name."""
    names: set[str] = set()
    for scenario in scenarios:
        validate(scenario)
        if scenario.name in names:
            raise ScenarioError(f"Two scenarios are called {scenario.name!r}.")
        names.add(scenario.name)

    if expected is not None and len(scenarios) != expected:
        raise ScenarioError(
            f"The dataset has {len(scenarios)} scenarios; {expected} were expected."
        )


def _validate_world(scenario: Scenario) -> None:
    world = scenario.world
    if not world.services:
        raise ScenarioError(f"{scenario.name}: a world needs at least one service.")
    for hours in world.hours:
        if not 0 <= hours.weekday <= 6:
            raise ScenarioError(
                f"{scenario.name}: weekday {hours.weekday} is not 0–6."
            )
        if hours.closes_at <= hours.opens_at:
            raise ScenarioError(
                f"{scenario.name}: business hours must close after they open."
            )

    refs = [seed.ref for seed in world.appointments]
    if len(refs) != len(set(refs)):
        raise ScenarioError(f"{scenario.name}: two seeded appointments share a ref.")
    for seed in world.appointments:
        _require_instant(scenario, seed.starts_at, f"seeded appointment {seed.ref!r}")


def _validate_tools(scenario: Scenario) -> None:
    for expected in scenario.expect.expected_tools:
        if expected.name not in TOOLS:
            raise ScenarioError(
                f"{scenario.name}: {expected.name!r} is not a tool. "
                f"Available: {', '.join(sorted(TOOLS))}."
            )
    for name in scenario.expect.forbidden_tools | scenario.expect.allow_tool_failures:
        if name not in TOOLS:
            raise ScenarioError(f"{scenario.name}: {name!r} is not a tool.")

    forbidden = scenario.expect.forbidden_tools
    for expected in scenario.expect.expected_tools:
        if expected.name in forbidden:
            raise ScenarioError(
                f"{scenario.name}: {expected.name!r} is both expected and forbidden."
            )


def _validate_claims(scenario: Scenario) -> None:
    for claim in scenario.expect.availability_claims:
        _require_instant(scenario, claim.starts_at, "an availability claim")
        if not claim.service.strip():
            raise ScenarioError(f"{scenario.name}: a claim needs a service name.")
        if not 0 <= claim.turn < len(scenario.script):
            raise ScenarioError(
                f"{scenario.name}: a claim names turn {claim.turn}, but the "
                f"script has {len(scenario.script)} turns."
            )


def _validate_appointments(scenario: Scenario) -> None:
    for spec in scenario.expect.final_appointments:
        _require_instant(scenario, spec.starts_at, "an expected appointment")
        if spec.status not in ("booked", "cancelled"):
            raise ScenarioError(
                f"{scenario.name}: {spec.status!r} is not an appointment status."
            )


def _validate_placeholders(scenario: Scenario) -> None:
    """Every `{{appointment:x}}` in the script must name a seeded appointment."""
    refs = scenario.world.refs
    for beat in scenario.script:
        for response in beat.responses:
            for use in response.tool_uses:
                arguments = use.arguments if isinstance(use.arguments, dict) else {}
                for value in arguments.values():
                    ref = placeholder_ref(value)
                    if ref is not None and ref not in refs:
                        raise ScenarioError(
                            f"{scenario.name}: the script refers to appointment "
                            f"{ref!r}, which the world does not seed."
                        )


def _require_instant(scenario: Scenario, value: Any, what: str) -> None:
    try:
        normalise_instant(value)
    except (ToolArgumentError, ValueError) as exc:
        raise ScenarioError(
            f"{scenario.name}: {what} has an unreadable time {value!r}."
        ) from exc


__all__ = [
    "EVAL_TIMEZONE",
    "KINDS",
    "TASKS",
    "AppointmentSpec",
    "Beat",
    "Claim",
    "ExpectedTool",
    "Expectation",
    "HoursSpec",
    "RealtimeExpectation",
    "Scenario",
    "ScenarioError",
    "SeedAppointment",
    "ServiceSpec",
    "World",
    "arguments_match",
    "normalise_argument",
    "normalise_instant",
    "placeholder_ref",
    "slot_key",
    "timezone",
    "validate",
    "validate_all",
    "weekdays",
]
