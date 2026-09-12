"""A call survives a refused calendar write — at every layer above it.

The defect these tests exist for was found by the milestone-9 evaluation
suite and diagnosed by measurement rather than by reading: a conflicting
reschedule raised the right domain error, and left the session in a
pending-rollback state that nothing reset. Every later statement on that
session then failed, so the caller was told nothing, the transcript was never
written, and the rest of the call was dead.

What was *not* wrong is worth stating, because the first diagnosis got it
backwards: the savepoint does its job. The appointment is expired and
restored, not left carrying times the database refused. The fault was the
transaction, not the object — and the fix is one rollback in
`CalendarService._write`.

Four layers, because a fix in one of them is not a fix for a call:
the calendar, the tool, the conversation, and the whole call.
"""

from datetime import UTC, datetime, time

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.calendar import CalendarService, SlotUnavailable
from app.config import Settings
from app.dialogue import Conversation
from app.models import Appointment, BusinessHours, Call, Service, Turn
from app.tools import ToolContext, book_appointment, cancel, reschedule

from .conftest import FakeModel, say, use_tools

MONDAY_TEN = datetime(2026, 3, 2, 10, tzinfo=UTC)
MONDAY_TWO = datetime(2026, 3, 2, 14, tzinfo=UTC)
ADA = ("Ada Lovelace", "+447700900123")
BEA = ("Bea Bramble", "+447700900456")


@pytest.fixture
def booked(session: Session, calendar_settings: Settings):
    """Two appointments for one staff member, so a move can collide."""
    session.add_all(
        [
            BusinessHours(weekday=day, opens_at=time(9), closes_at=time(17))
            for day in range(5)
        ]
    )
    service = Service(name="Haircut", duration_minutes=30, staff_id="sam")
    session.add(service)
    session.commit()

    calendar = CalendarService(session, calendar_settings)
    mine = calendar.book(
        service_id=service.id,
        starts_at=MONDAY_TEN,
        customer_name=ADA[0],
        phone=ADA[1],
    )
    theirs = calendar.book(
        service_id=service.id,
        starts_at=MONDAY_TWO,
        customer_name=BEA[0],
        phone=BEA[1],
    )
    return calendar, service, mine, theirs


# --- layer one: the calendar ----------------------------------------------


def test_a_conflicting_reschedule_still_raises(session: Session, booked) -> None:
    """The domain error is unchanged. Only the session state is."""
    calendar, _service, mine, _theirs = booked

    with pytest.raises(SlotUnavailable):
        calendar.reschedule(mine.id, MONDAY_TWO)


def test_the_session_is_usable_afterwards(session: Session, booked) -> None:
    """The whole of the defect, in one assertion."""
    calendar, _service, mine, _theirs = booked

    with pytest.raises(SlotUnavailable):
        calendar.reschedule(mine.id, MONDAY_TWO)

    assert session.is_active is True


def test_a_query_works_afterwards(session: Session, booked) -> None:
    calendar, _service, mine, _theirs = booked

    with pytest.raises(SlotUnavailable):
        calendar.reschedule(mine.id, MONDAY_TWO)

    assert len(session.execute(select(Appointment)).scalars().all()) == 2


def test_the_appointment_keeps_its_original_time(session: Session, booked) -> None:
    """In memory and in the database. The savepoint was never the problem."""
    calendar, _service, mine, _theirs = booked

    with pytest.raises(SlotUnavailable):
        calendar.reschedule(mine.id, MONDAY_TWO)

    assert mine.starts_at == MONDAY_TEN
    assert inspect(mine).modified is False


def test_a_later_reschedule_to_a_free_time_still_works(
    session: Session, booked
) -> None:
    """The session is not merely usable; it is correct."""
    calendar, _service, mine, _theirs = booked
    free = datetime(2026, 3, 2, 15, tzinfo=UTC)

    with pytest.raises(SlotUnavailable):
        calendar.reschedule(mine.id, MONDAY_TWO)

    assert calendar.reschedule(mine.id, free).starts_at == free


def test_a_refused_booking_also_leaves_the_session_usable(
    session: Session, booked
) -> None:
    """The insert path was never broken. It is asserted so it stays that way."""
    calendar, service, _mine, _theirs = booked

    with pytest.raises(SlotUnavailable):
        calendar.book(
            service_id=service.id,
            starts_at=MONDAY_TEN,
            customer_name=BEA[0],
            phone=BEA[1],
        )

    assert session.is_active is True
    session.add(Call(from_number=ADA[1], to_number="+441234567890"))
    session.commit()


def test_a_cancellation_works_after_a_refused_reschedule(
    session: Session, booked
) -> None:
    calendar, _service, mine, _theirs = booked

    with pytest.raises(SlotUnavailable):
        calendar.reschedule(mine.id, MONDAY_TWO)

    assert calendar.cancel(mine.id).status.value == "cancelled"


# --- layer two: the tool ---------------------------------------------------


def test_the_tool_reports_the_conflict(session: Session, booked, tools) -> None:
    _calendar, _service, mine, _theirs = booked

    result = reschedule(
        tools, appointment_id=mine.id, new_starts_at=MONDAY_TWO.isoformat()
    )

    assert result.success is False
    assert result.data["slot_taken"] is True


def test_another_tool_works_after_a_failed_reschedule(
    session: Session, booked, tools
) -> None:
    """A failed tool must not take the next one with it."""
    _calendar, _service, mine, theirs = booked

    reschedule(tools, appointment_id=mine.id, new_starts_at=MONDAY_TWO.isoformat())
    after = cancel(tools, appointment_id=theirs.id)

    assert after.success is True


def test_booking_works_after_a_failed_reschedule(
    session: Session, booked, tools
) -> None:
    _calendar, _service, mine, _theirs = booked

    reschedule(tools, appointment_id=mine.id, new_starts_at=MONDAY_TWO.isoformat())
    after = book_appointment(
        tools,
        service_name="Haircut",
        starts_at="2026-03-02T15:00:00+00:00",
        customer_name="Cai Rivers",
        phone="+447700900789",
    )

    assert after.success is True


# --- layer three: the conversation ----------------------------------------


def test_the_transcript_is_written_after_a_failed_reschedule(
    session: Session, booked, call: Call, calendar_settings: Settings
) -> None:
    """This is the turn the caller never used to hear."""
    _calendar, _service, mine, _theirs = booked
    conversation = Conversation(
        session,
        call,
        FakeModel(
            use_tools(
                (
                    "reschedule",
                    {
                        "appointment_id": str(mine.id),
                        "new_starts_at": MONDAY_TWO.isoformat(),
                    },
                )
            ),
            say("Somebody has that time, I'm afraid."),
        ),
        calendar_settings,
    )

    result = conversation.send("Move my haircut to two o'clock.")

    assert result.text == "Somebody has that time, I'm afraid."
    assert result.failed is False
    assert result.tool_calls[0].success is False
    assert len(session.execute(select(Turn)).scalars().all()) == 2


def test_a_second_turn_works_after_a_failed_reschedule(
    session: Session, booked, call: Call, calendar_settings: Settings
) -> None:
    """The rest of the call used to be dead. It is not."""
    _calendar, _service, mine, _theirs = booked
    conversation = Conversation(
        session,
        call,
        FakeModel(
            use_tools(
                (
                    "reschedule",
                    {
                        "appointment_id": str(mine.id),
                        "new_starts_at": MONDAY_TWO.isoformat(),
                    },
                )
            ),
            say("That time has gone."),
            say("Of course — we're open until five."),
        ),
        calendar_settings,
    )

    conversation.send("Move it to two.")
    second = conversation.send("Never mind, I'll ring back.")

    assert second.text == "Of course — we're open until five."
    assert len(session.execute(select(Turn)).scalars().all()) == 4


def test_the_call_can_still_be_ended(
    session: Session, booked, call: Call, calendar_settings: Settings
) -> None:
    """`ended_at` used to be lost with everything else."""
    _calendar, _service, mine, _theirs = booked
    conversation = Conversation(
        session,
        call,
        FakeModel(
            use_tools(
                (
                    "reschedule",
                    {
                        "appointment_id": str(mine.id),
                        "new_starts_at": MONDAY_TWO.isoformat(),
                    },
                )
            ),
            say("That time has gone."),
        ),
        calendar_settings,
    )
    conversation.send("Move it to two.")

    call.ended_at = datetime.now(UTC)
    session.commit()

    assert session.get(Call, call.id).ended_at is not None


def test_cost_can_still_be_recorded_after_a_failed_reschedule(
    session: Session, booked, call: Call, cost_settings: Settings
) -> None:
    """Milestone 8's recorder needs a usable session, like everything else."""
    from app.cost import record_turn_cost
    from app.models import CallCost

    _calendar, _service, mine, _theirs = booked
    conversation = Conversation(
        session,
        call,
        FakeModel(
            use_tools(
                (
                    "reschedule",
                    {
                        "appointment_id": str(mine.id),
                        "new_starts_at": MONDAY_TWO.isoformat(),
                    },
                )
            ),
            say("That time has gone."),
        ),
        cost_settings,
    )
    result = conversation.send("Move it to two.")

    class _NoSpeech:
        stt_provider = ""
        stt_audio_ms = None
        tts_provider = ""
        tts_characters = None

    record_turn_cost(
        session, result, _NoSpeech(), cost_settings, llm_provider="anthropic"
    )

    # The model reported no usage here, so no row is the correct outcome —
    # what matters is that the recorder could run at all.
    assert session.execute(select(CallCost)).scalars().all() == []
    assert session.is_active is True
