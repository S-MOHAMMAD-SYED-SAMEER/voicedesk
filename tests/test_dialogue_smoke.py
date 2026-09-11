"""A whole call, end to end, with a scripted model and a real database.

What is checked here is what no single component's tests can check: that a
caller can be booked in without a phone, and that nothing anywhere in the
path can tell them something that did not happen.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dialogue import MODEL_FAILURE_REPLY, Conversation
from app.models import Appointment, AppointmentStatus, Call, Service, Turn
from app.providers.llm import ModelUnavailable

from .conftest import FakeModel, say, use_tools

MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"


def _check(service: str = "Haircut", day: str = MONDAY):
    return ("check_availability", {"service_name": service, "day": day})


def _book(starts_at: str = TEN, name: str = "Ada Lovelace"):
    return (
        "book_appointment",
        {
            "service_name": "Haircut",
            "starts_at": starts_at,
            "customer_name": name,
            "phone": "+447700900123",
        },
    )


# --- a caller gets booked in ----------------------------------------------


def test_a_caller_is_booked_in_without_a_phone_anywhere_in_sight(
    session: Session, dialogue, call: Call, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(
        say("Of course — what day suits you?"),
        use_tools(_check()),
        say("I have nine, half nine or ten. Which would you like?"),
        use_tools(_book()),
        say("You're booked in for ten o'clock on Monday, Ada."),
    )

    conversation.send("I'd like a haircut please.")
    conversation.send("Monday, if you have anything.")
    result = conversation.send("Ten o'clock, and it's Ada Lovelace.")

    appointment = session.execute(select(Appointment)).scalar_one()
    assert str(appointment.id) == result.booked_appointment_id
    assert appointment.customer_name == "Ada Lovelace"
    assert appointment.status is AppointmentStatus.BOOKED
    assert appointment.call_id == call.id
    assert len(session.execute(select(Turn)).scalars().all()) == 6


def test_a_booking_is_attributed_to_the_call_it_was_made_on(
    session: Session, dialogue, call: Call, open_weekdays, haircut: Service
) -> None:
    """The call comes from the conversation, never from the model."""
    dialogue(use_tools(_check()), use_tools(_book()), say("Done.")).send(
        "Haircut Monday at ten, Ada Lovelace."
    )

    appointment = session.execute(select(Appointment)).scalar_one()
    assert appointment.call_id == call.id


# --- nothing may be claimed that did not happen ---------------------------


def test_the_model_cannot_book_a_time_the_calendar_never_offered(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """The hallucinated-availability rate is zero by construction, not by hope."""
    result = dialogue(
        use_tools(_book(starts_at="2026-03-02T13:37:00+00:00")),
        say("Let me just check that for you."),
    ).send("Book me in at twenty to two.")

    assert result.booked_appointment_id is None
    assert result.tool_calls[0].success is False
    assert session.execute(select(Appointment)).first() is None


def test_a_failed_booking_is_reported_as_failed_however_the_model_phrases_it(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """The model says it went through; the record says otherwise, and wins."""
    result = dialogue(
        use_tools(_check()),
        use_tools(_book(starts_at="2026-03-02T20:00:00+00:00")),
        say("Lovely, you're all booked in for eight this evening."),
    ).send("Eight in the evening, please.")

    assert result.booked_appointment_id is None
    assert result.tool_calls[-1].success is False
    assert session.execute(select(Appointment)).first() is None


def test_losing_a_race_reaches_the_model_as_a_recoverable_failure(
    session: Session, dialogue, calendar, open_weekdays, haircut: Service
) -> None:
    """Somebody else takes the slot between the check and the booking.

    This is the case an availability check cannot cover, and the reason the
    calendar writes and lets the database refuse rather than checking first.
    The model has to be told, in a way it can act on.
    """
    from datetime import UTC, datetime

    class RacesUs:
        """Books the slot itself, after availability has been computed."""

        def __init__(self) -> None:
            self.calls = 0
            self.requests: list[dict] = []

        def respond(self, *, system, messages, tools):
            self.calls += 1
            self.requests.append({"messages": list(messages)})
            if self.calls == 1:
                return use_tools(_check())
            if self.calls == 2:
                calendar.book(
                    service_id=haircut.id,
                    starts_at=datetime(2026, 3, 2, 10, tzinfo=UTC),
                    customer_name="Grace Hopper",
                    phone="+447700900999",
                )
                return use_tools(_book())
            return say("I'm sorry, that one has just gone.")

    model = RacesUs()

    result = dialogue(model=model).send("Ten o'clock please.")

    assert result.text == "I'm sorry, that one has just gone."
    assert result.booked_appointment_id is None
    assert result.tool_calls[-1].data["slot_taken"] is True

    block = model.requests[-1]["messages"][-1].content[0]
    assert block["is_error"] is True
    assert "slot_taken" in block["content"]

    booked = (
        session.execute(
            select(Appointment).filter_by(status=AppointmentStatus.BOOKED)
        )
        .scalars()
        .all()
    )
    assert [appointment.customer_name for appointment in booked] == ["Grace Hopper"]


def test_a_model_outage_never_produces_a_booking(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    result = dialogue(raises=ModelUnavailable("the api fell over")).send(
        "Book me a haircut."
    )

    assert result.text == MODEL_FAILURE_REPLY
    assert result.failed is True
    assert session.execute(select(Appointment)).first() is None


# --- the layering the specification asks for ------------------------------


def test_the_dialogue_layer_needs_only_a_session_a_call_and_a_model(
    session: Session, call: Call, calendar_settings, open_weekdays, haircut: Service
) -> None:
    """"If testing a booking flow requires a phone call, the layering is wrong.\""""
    conversation = Conversation(
        session, call, FakeModel(say("Certainly.")), calendar_settings
    )

    assert conversation.send("Hello?").text == "Certainly."


def test_a_conversation_reads_its_menu_once_at_the_start(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """The menu is what the business offers, not something that shifts mid-call."""
    conversation = dialogue(say("One moment."), say("Still here."))
    before = conversation.system_prompt

    session.add(Service(name="Beard trim", duration_minutes=15, staff_id="sam"))
    session.commit()
    conversation.send("Hello?")

    assert conversation.system_prompt == before
    assert "Beard trim" not in conversation.system_prompt
