"""What the tool layer must be, taken as a whole.

Three things are checked here that no single tool's tests can check: that the
tools delegate rather than reimplement, that a whole call's worth of tool use
hangs together, and that the database still settles a race when two callers
reach it through the tools.
"""

import ast
import pathlib
from datetime import UTC, datetime

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Appointment, AppointmentStatus, Service
from app.tools import (
    ToolContext,
    book_appointment,
    cancel,
    check_availability,
    reschedule,
    take_message,
    transfer_to_human,
)

TOOLS_DIRECTORY = pathlib.Path(__file__).resolve().parent.parent / "app" / "tools"


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC)


# --- the tools must delegate, not reimplement ------------------------------

# Any of these appearing in a tool module would mean the calendar had been
# reimplemented alongside itself, and two answers to "is this free?" is one
# too many. `Service` is exempt: resolving a service *name* to the row is the
# tool layer's own job, because the calendar takes ids.
FORBIDDEN = (
    "BusinessHours",
    "tstzrange",
    "select(Appointment",
    "AppointmentStatus",
    "timedelta",
    "duration_minutes",
    "candidate_slots",
    "free_slots",
    "periods_for",
)


def test_no_tool_module_reimplements_the_calendar() -> None:
    for module in sorted(TOOLS_DIRECTORY.glob("*.py")):
        source = module.read_text()
        for forbidden in FORBIDDEN:
            assert forbidden not in source, f"{module.name} contains {forbidden!r}"


def test_the_tool_layer_reaches_the_calendar_only_through_the_service() -> None:
    """No tool imports the calendar's internals behind `CalendarService`."""
    for module in sorted(TOOLS_DIRECTORY.glob("*.py")):
        source = module.read_text()
        assert "app.calendar.availability" not in source, module.name
        assert "app.calendar.hours" not in source, module.name


def _imported_modules(source: str) -> set[str]:
    """Top-level package names a module imports, from its parsed syntax."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def test_no_tool_imports_a_model_provider_or_telephony() -> None:
    """Milestone 3 is callable without audio, Twilio or an API key.

    Parsed rather than grepped, so a docstring may name the milestone that
    will do the transferring without tripping the check.
    """
    forbidden = {"anthropic", "openai", "twilio", "websockets", "google"}
    for module in sorted(TOOLS_DIRECTORY.glob("*.py")):
        assert not _imported_modules(module.read_text()) & forbidden, module.name


# --- a whole call's worth of tool use --------------------------------------


def test_a_caller_books_moves_and_cancels(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    availability = check_availability(tools, service_name="Haircut", day="2026-03-02")
    first_free = availability.data["slots"][0]["starts_at"]

    booked = book_appointment(
        tools,
        service_name="Haircut",
        starts_at=first_free,
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )
    assert booked.success, booked.error

    moved = reschedule(
        tools,
        appointment_id=booked.data["appointment_id"],
        new_starts_at=_at(14).isoformat(),
    )
    assert moved.success, moved.error

    cancelled = cancel(tools, appointment_id=booked.data["appointment_id"])
    assert cancelled.success, cancelled.error
    assert cancelled.data["status"] == "cancelled"


def test_a_caller_the_calendar_cannot_help_is_not_left_stranded(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    """The two exits that exist when booking is not the answer."""
    message = take_message(
        tools,
        caller_name="Ada Lovelace",
        phone="+447700900123",
        message="I need to talk about a refund.",
    )
    escalation = transfer_to_human(tools, reason="Refund request, needs a human.")

    assert message.success and escalation.success
    assert escalation.data["escalated"] is True


def test_a_booking_offered_by_check_availability_is_always_bookable(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    """Every slot the tool offers is real: the hallucination rate is zero."""
    slots = check_availability(tools, service_name="Haircut", day="2026-03-02").data[
        "slots"
    ]

    for index, slot in enumerate(slots[:4]):
        result = book_appointment(
            tools,
            service_name="Haircut",
            starts_at=slot["starts_at"],
            customer_name=f"Caller {index}",
            phone="+447700900123",
        )
        # Later slots on the 15-minute grid overlap earlier 30-minute
        # bookings, so only a clash is acceptable — never "no such time".
        assert result.success or result.data.get("slot_taken"), result.error


# --- the race, through the tools -------------------------------------------


def test_two_callers_racing_through_the_tools_and_only_one_wins(
    migrated_engine: Engine,
    session: Session,
    calendar_settings: Settings,
    open_weekdays,
    haircut: Service,
) -> None:
    """The constraint still arbitrates when the tools are the front door.

    Both contexts see the slot free, both book, and the loser gets a failed
    result rather than an exception — which is what lets a dialogue recover.
    """
    with Session(migrated_engine) as first, Session(migrated_engine) as second:
        alice = ToolContext(session=first, settings=calendar_settings)
        bob = ToolContext(session=second, settings=calendar_settings)

        offered = "2026-03-02T10:00:00+00:00"
        for context in (alice, bob):
            starts = [
                slot["starts_at"]
                for slot in check_availability(
                    context, service_name="Haircut", day="2026-03-02"
                ).data["slots"]
            ]
            assert offered in starts

        results = [
            book_appointment(
                context,
                service_name="Haircut",
                starts_at=offered,
                customer_name=name,
                phone="+447700900123",
            )
            for context, name in ((alice, "Alice"), (bob, "Bob"))
        ]

    assert [result.success for result in results] == [True, False]
    assert results[1].data["slot_taken"] is True

    booked = (
        session.execute(
            select(Appointment).filter_by(status=AppointmentStatus.BOOKED)
        )
        .scalars()
        .all()
    )
    assert len(booked) == 1
    assert booked[0].customer_name == "Alice"


def test_a_loser_of_a_race_can_still_use_its_session(
    migrated_engine: Engine,
    calendar_settings: Settings,
    open_weekdays,
    haircut: Service,
) -> None:
    """A refused write must not poison the transaction the dialogue is using."""
    with Session(migrated_engine) as first, Session(migrated_engine) as second:
        alice = ToolContext(session=first, settings=calendar_settings)
        bob = ToolContext(session=second, settings=calendar_settings)

        for context, name in ((alice, "Alice"), (bob, "Bob")):
            book_appointment(
                context,
                service_name="Haircut",
                starts_at="2026-03-02T10:00:00+00:00",
                customer_name=name,
                phone="+447700900123",
            )

        recovery = book_appointment(
            bob,
            service_name="Haircut",
            starts_at="2026-03-02T11:00:00+00:00",
            customer_name="Bob",
            phone="+447700900123",
        )

    assert recovery.success, recovery.error
