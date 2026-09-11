"""The transcript: what a call leaves behind in `turns` and `tool_calls`."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Call, Service, ToolCall, Turn, TurnRole
from app.providers.llm import ModelUnavailable

from .conftest import FakeModel, say, use_tools

MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"


def _check(service: str = "Haircut"):
    return ("check_availability", {"service_name": service, "day": MONDAY})


def _book():
    return (
        "book_appointment",
        {
            "service_name": "Haircut",
            "starts_at": TEN,
            "customer_name": "Ada Lovelace",
            "phone": "+447700900123",
        },
    )


def _turns(session: Session) -> list[Turn]:
    return list(
        session.execute(select(Turn).order_by(Turn.created_at)).scalars().all()
    )


def test_a_turn_writes_one_caller_row_and_one_agent_row(
    session: Session, dialogue, call: Call, open_weekdays, haircut: Service
) -> None:
    dialogue(say("We close at five.")).send("What time do you close?")

    turns = _turns(session)
    assert [turn.role for turn in turns] == [TurnRole.CALLER, TurnRole.AGENT]
    assert [turn.text for turn in turns] == [
        "What time do you close?",
        "We close at five.",
    ]
    assert all(turn.call_id == call.id for turn in turns)


def test_the_caller_turn_is_ordered_before_the_agent_turn(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """Both rows are written together, so the timestamps are set explicitly."""
    dialogue(say("Certainly.")).send("Hello?")

    caller, agent = _turns(session)
    assert caller.created_at < agent.created_at


def test_turns_accumulate_across_a_conversation(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(say("Hello."), say("Until five."))
    conversation.send("Hi.")
    conversation.send("Closing time?")

    turns = _turns(session)
    assert [turn.role for turn in turns] == [
        TurnRole.CALLER,
        TurnRole.AGENT,
        TurnRole.CALLER,
        TurnRole.AGENT,
    ]


def test_no_empty_turn_is_written_for_a_tool_only_model_response(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """One agent turn per send, whatever happened inside it."""
    dialogue(use_tools(_check()), use_tools(_book()), say("Booked.")).send(
        "Haircut Monday please."
    )

    assert len(_turns(session)) == 2


def test_model_latency_is_recorded_on_the_agent_turn(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    dialogue(
        use_tools(_check(), latency_ms=30), say("Ten is free.", latency_ms=20)
    ).send("Monday?")

    caller, agent = _turns(session)
    assert agent.llm_latency_ms == 50
    assert caller.llm_latency_ms is None


def test_speech_columns_stay_empty(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """Milestone 4 has no audio, so it invents no audio measurements."""
    dialogue(say("Certainly.")).send("Hello?")

    for turn in _turns(session):
        assert turn.audio_ms is None
        assert turn.stt_latency_ms is None
        assert turn.tts_latency_ms is None


# --- tool calls -----------------------------------------------------------


def test_tool_calls_hang_off_the_agent_turn(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    dialogue(use_tools(_check()), say("Ten is free.")).send("Monday?")

    caller, agent = _turns(session)
    assert [row.tool_name for row in agent.tool_calls] == ["check_availability"]
    assert caller.tool_calls == []


def test_a_successful_call_stores_its_result(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    dialogue(use_tools(_check()), say("Ten is free.")).send("Monday?")

    row = session.execute(select(ToolCall)).scalar_one()
    assert row.success is True
    assert row.error is None
    assert row.result["service"] == "Haircut"
    assert row.result["slots"][0]["starts_at"] == "2026-03-02T09:00:00+00:00"
    assert TEN in [slot["starts_at"] for slot in row.result["slots"]]


def test_the_arguments_are_stored_verbatim(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    dialogue(use_tools(_check()), say("Ten is free.")).send("Monday?")

    row = session.execute(select(ToolCall)).scalar_one()
    assert row.arguments == {"service_name": "Haircut", "day": MONDAY}


def test_a_failed_call_stores_a_null_result_and_its_error(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """A failed call has no result to record, only an error."""
    dialogue(use_tools(_check(service="Hovercraft")), say("Only haircuts.")).send(
        "A hovercraft?"
    )

    row = session.execute(select(ToolCall)).scalar_one()
    assert row.success is False
    assert row.result is None
    assert "Hovercraft" in row.error


def test_recovery_data_from_a_failure_is_not_persisted_as_a_result(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """It went to the model in the tool_result block; it is not an outcome."""
    dialogue(use_tools(_check(service="Hovercraft")), say("Only haircuts.")).send(
        "A hovercraft?"
    )

    row = session.execute(select(ToolCall)).scalar_one()
    assert row.result is None


def test_a_blocked_booking_is_persisted_as_a_failed_call(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    dialogue(use_tools(_book()), say("Let me check that first.")).send("Ten please.")

    row = session.execute(select(ToolCall)).scalar_one()
    assert row.tool_name == "book_appointment"
    assert row.success is False
    assert row.result is None
    assert "not been verified" in row.error


def test_an_unknown_tool_is_persisted_so_it_can_be_counted(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """Wrong-tool-call rate is a metric; an unrecorded attempt is invisible."""
    dialogue(use_tools(("order_a_taxi", {})), say("I can't do that.")).send(
        "Get me a cab."
    )

    row = session.execute(select(ToolCall)).scalar_one()
    assert row.tool_name == "order_a_taxi"
    assert row.success is False


def test_latency_is_recorded_per_tool_call(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    dialogue(use_tools(_check()), say("Ten is free.")).send("Monday?")

    row = session.execute(select(ToolCall)).scalar_one()
    assert row.latency_ms is not None and row.latency_ms >= 0


def test_every_call_in_one_turn_is_stored(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    dialogue(use_tools(_check(), _check(service="Hovercraft")), say("Monday?")).send(
        "Monday?"
    )

    caller, agent = _turns(session)
    assert len(agent.tool_calls) == 2
    assert {row.success for row in agent.tool_calls} == {True, False}


# --- failed turns ----------------------------------------------------------


def test_a_failed_turn_is_still_persisted(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """The call that went wrong is the one somebody will want to read."""
    result = dialogue(raises=ModelUnavailable("the api fell over")).send("Monday?")

    caller, agent = _turns(session)
    assert caller.text == "Monday?"
    assert agent.text == result.text
    assert agent.tool_calls == []


def test_a_cut_off_turn_keeps_the_tool_calls_it_made(
    session: Session, dialogue, calendar_settings, open_weekdays, haircut: Service
) -> None:
    settings = calendar_settings.model_copy(update={"max_tool_iterations": 2})
    model = FakeModel(*[use_tools(_check()) for _ in range(5)])

    dialogue(model=model, settings=settings).send("Monday?")

    caller, agent = _turns(session)
    assert len(agent.tool_calls) == 2


def test_deleting_a_call_takes_its_transcript_with_it(
    session: Session, dialogue, call: Call, open_weekdays, haircut: Service
) -> None:
    dialogue(use_tools(_check()), say("Ten is free.")).send("Monday?")

    session.delete(call)
    session.commit()

    assert session.execute(select(Turn)).first() is None
    assert session.execute(select(ToolCall)).first() is None
