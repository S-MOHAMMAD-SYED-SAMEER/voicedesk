"""Building the business a scenario happens in.

Services, opening hours and any appointments that already exist, written into
the evaluation database before the phone rings.

Seeded appointments go in through `CalendarService.book` rather than through a
raw insert. That costs a little and buys a lot: the fixture is subject to the
same opening-hours rules and the same exclusion constraint as a real booking,
so a scenario cannot quietly set up a world the application could never have
produced. A fixture that the calendar refuses is a dataset fault and says so.

Nothing here computes availability, durations or conflicts. There is one
calendar in this repository and this is not it.
"""

import uuid

from sqlalchemy.orm import Session

from app.calendar import CalendarError, CalendarService
from app.config import Settings
from app.evals.scenario import (
    Scenario,
    ScenarioError,
    World,
    normalise_instant,
    placeholder_ref,
)
from app.models import (
    AppointmentStatus,
    BusinessHours,
    Call,
    CallDirection,
    Service,
)


def build(session: Session, world: World, settings: Settings) -> dict[str, uuid.UUID]:
    """Write the world, and return the identifier of each seeded appointment.

    The returned mapping is keyed by the scenario's own `ref`, which is what a
    script writes before any identifier exists.
    """
    services = _services(session, world)
    _hours(session, world)
    return _appointments(session, world, services, settings)


def start_call(session: Session, world: World) -> Call:
    """The `calls` row every scenario's conversation is attached to."""
    call = Call(
        direction=CallDirection.INBOUND,
        from_number=world.from_number,
        to_number=world.to_number,
    )
    session.add(call)
    session.commit()
    return call


def resolve(value: object, appointments: dict[str, uuid.UUID]) -> object:
    """Swap `{{appointment:ref}}` for the identifier it stands in for."""
    ref = placeholder_ref(value)
    if ref is None:
        return value
    try:
        return str(appointments[ref])
    except KeyError:
        raise ScenarioError(
            f"The script refers to appointment {ref!r}, which was not seeded."
        ) from None


def resolve_arguments(
    arguments: object, appointments: dict[str, uuid.UUID]
) -> dict[str, object]:
    """Every placeholder in one tool call's arguments, substituted."""
    if not isinstance(arguments, dict):
        return {}
    return {key: resolve(value, appointments) for key, value in arguments.items()}


def scripted_responses(scenario: Scenario, appointments: dict[str, uuid.UUID]):
    """The scenario's responses with every appointment placeholder resolved.

    `ModelResponse` is frozen and its `raw_content` is what the dialogue layer
    replays to the model, so both the parsed `tool_uses` and the raw blocks
    have to be substituted or the two would describe different calls.
    """
    import dataclasses

    from app.providers.llm import ToolUse

    resolved = []
    for response in scenario.responses:
        uses = tuple(
            ToolUse(
                id=use.id,
                name=use.name,
                arguments=resolve_arguments(use.arguments, appointments),
            )
            for use in response.tool_uses
        )
        content = [
            (
                {**block, "input": resolve_arguments(block.get("input"), appointments)}
                if block.get("type") == "tool_use"
                else block
            )
            for block in response.raw_content
        ]
        resolved.append(
            dataclasses.replace(
                response, tool_uses=list(uses), raw_content=content
            )
        )
    return resolved


# --- internals -------------------------------------------------------------


def _services(session: Session, world: World) -> dict[str, Service]:
    built: dict[str, Service] = {}
    rows = []
    for spec in world.services:
        row = Service(
            name=spec.name,
            duration_minutes=spec.duration_minutes,
            staff_id=spec.staff_id,
            active=spec.active,
        )
        rows.append(row)
        # An ambiguous-service scenario seeds two rows with one name on
        # purpose; the second simply wins this lookup, and nothing reads it.
        built[spec.name.casefold()] = row
    session.add_all(rows)
    session.commit()
    return built


def _hours(session: Session, world: World) -> None:
    session.add_all(
        [
            BusinessHours(
                weekday=spec.weekday,
                opens_at=spec.opens_at,
                closes_at=spec.closes_at,
            )
            for spec in world.hours
        ]
    )
    session.commit()


def _appointments(
    session: Session,
    world: World,
    services: dict[str, Service],
    settings: Settings,
) -> dict[str, uuid.UUID]:
    if not world.appointments:
        return {}

    calendar = CalendarService(session, settings)
    built: dict[str, uuid.UUID] = {}

    for seed in world.appointments:
        service = services.get(seed.service_name.casefold())
        if service is None:
            raise ScenarioError(
                f"A seeded appointment names service {seed.service_name!r}, "
                "which the world does not offer."
            )
        try:
            appointment = calendar.book(
                service_id=service.id,
                starts_at=normalise_instant(seed.starts_at),
                customer_name=seed.customer_name,
                phone=seed.phone,
            )
        except CalendarError as exc:
            raise ScenarioError(
                f"The calendar refused the seeded appointment {seed.ref!r}: "
                f"{exc}. A fixture the application could not have produced is "
                "not a fixture."
            ) from exc

        if seed.cancelled:
            calendar.cancel(appointment.id)
        built[seed.ref] = appointment.id

    _expunge(session)
    return built


def _expunge(session: Session) -> None:
    """Drop the identity map so later reads see the database, not a cache."""
    session.expire_all()


def cancelled_status() -> AppointmentStatus:
    return AppointmentStatus.CANCELLED


__all__ = [
    "build",
    "resolve",
    "resolve_arguments",
    "scripted_responses",
    "start_call",
]
