"""The evaluation dataset: eighteen calls, fixed in the repository.

    booking.py    seven ways a booking goes, four of them adversarial
    changes.py    moving and cancelling what already exists
    handover.py   taking a message, and the four escalation triggers
    streaming.py  the two that need the realtime path

Deterministic, versioned, and entirely invented. Fixed dates, invented names,
numbers from the UK's reserved drama range: no real personal information
appears anywhere here, and none should ever be added.

The dataset is validated before anything runs — names unique, tools real,
claims readable, expectations well formed — so a malformed scenario is caught
as a dataset fault at load time rather than mistaken at run time for the
receptionist misbehaving.
"""

from app.evals.dataset import booking, changes, handover, streaming
from app.evals.scenario import Scenario, ScenarioError, validate_all

# The specification asks for at least fifteen. Eighteen is what this dataset
# holds, and the count is asserted so that losing one is a loud failure.
EXPECTED_SCENARIOS = 18

# The families the dataset has to cover, so that a scenario deleted from one
# of them cannot go unnoticed.
FAMILIES = {
    "booking": ("booking_available", "booking_caller_changes_mind"),
    "unavailable": (
        "booking_unavailable_slot",
        "booking_slot_taken_between_check_and_book",
    ),
    "unknown_service": ("booking_unknown_service",),
    "ambiguity": ("booking_ambiguous_service",),
    "missing_details": ("booking_missing_details",),
    "reschedule": ("reschedule_success", "reschedule_conflict"),
    "cancel": ("cancel_success", "cancel_already_cancelled"),
    "message": ("take_message",),
    "escalation": (
        "escalate_explicit_request",
        "escalate_medical",
        "escalate_legal",
        "escalate_complaint_refund",
    ),
    "barge_in": ("realtime_barge_in_booking_survives",),
    "silence": ("realtime_silence_asks_no_model",),
}

SCENARIOS: tuple[Scenario, ...] = (
    *booking.SCENARIOS,
    *changes.SCENARIOS,
    *handover.SCENARIOS,
    *streaming.SCENARIOS,
)


def all_scenarios() -> tuple[Scenario, ...]:
    """Every scenario, validated.

    Validation runs here rather than at import so that a broken dataset
    produces a clear message from the runner instead of an import error from
    somewhere in the middle of a package.
    """
    validate_all(SCENARIOS, expected=EXPECTED_SCENARIOS)
    _validate_families()
    return SCENARIOS


def by_name(names: tuple[str, ...]) -> tuple[Scenario, ...]:
    """The named scenarios, in dataset order, or a clear refusal."""
    known = {scenario.name: scenario for scenario in all_scenarios()}
    missing = [name for name in names if name not in known]
    if missing:
        raise ScenarioError(
            f"No scenario called {', '.join(missing)}. "
            f"Available: {', '.join(sorted(known))}."
        )
    wanted = set(names)
    return tuple(
        scenario for scenario in SCENARIOS if scenario.name in wanted
    )


def names() -> tuple[str, ...]:
    return tuple(scenario.name for scenario in SCENARIOS)


def _validate_families() -> None:
    present = {scenario.name for scenario in SCENARIOS}
    for family, members in FAMILIES.items():
        missing = [name for name in members if name not in present]
        if missing:
            raise ScenarioError(
                f"The {family!r} family is missing {', '.join(missing)}."
            )

    covered = {name for members in FAMILIES.values() for name in members}
    uncovered = present - covered
    if uncovered:
        raise ScenarioError(
            f"These scenarios belong to no declared family: "
            f"{', '.join(sorted(uncovered))}."
        )


__all__ = [
    "EXPECTED_SCENARIOS",
    "FAMILIES",
    "SCENARIOS",
    "all_scenarios",
    "by_name",
    "names",
]
