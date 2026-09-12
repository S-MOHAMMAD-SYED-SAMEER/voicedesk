"""The turn manager: the loop, its limits, and what it says when it fails."""

import json
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.dialogue import (
    LOOP_EXHAUSTED_REPLY,
    MODEL_FAILURE_REPLY,
    SYSTEM_PROMPT_VERSION,
)
from app.models import Appointment, Service
from app.providers.llm import ModelRefused, ModelUnavailable

from .conftest import FakeModel, say, use_tools, used

MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"


def _check(day: str = MONDAY, service: str = "Haircut"):
    return ("check_availability", {"service_name": service, "day": day})


def _book(starts_at: str = TEN, service: str = "Haircut"):
    return (
        "book_appointment",
        {
            "service_name": service,
            "starts_at": starts_at,
            "customer_name": "Ada Lovelace",
            "phone": "+447700900123",
        },
    )


# --- plain conversation ---------------------------------------------------


def test_a_reply_with_no_tools_comes_straight_back(
    dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(say("We're open until five."))

    result = conversation.send("What time do you close?")

    assert result.text == "We're open until five."
    assert result.tool_calls == []
    assert result.failed is False
    assert result.prompt_version == SYSTEM_PROMPT_VERSION


def test_the_model_is_given_the_system_prompt_and_the_tools(
    dialogue, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("Certainly."))
    conversation = dialogue(model=model)

    conversation.send("Hello?")

    request = model.requests[0]
    assert "- Haircut" in request["system"]
    assert request["system"] == conversation.system_prompt
    assert [tool.name for tool in request["tools"]] == [
        "check_availability",
        "book_appointment",
        "reschedule",
        "cancel",
        "take_message",
        "transfer_to_human",
    ]


def test_history_grows_across_turns(
    dialogue, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("Hello."), say("Until five."))
    conversation = dialogue(model=model)

    conversation.send("Hi.")
    conversation.send("What time do you close?")

    assert [message.role for message in conversation.history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert len(model.requests[1]["messages"]) == 3


# --- tool use -------------------------------------------------------------


def test_an_availability_request_runs_the_tool_and_returns_the_reply(
    dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(
        use_tools(_check()), say("I can do nine, half nine or ten.")
    )

    result = conversation.send("Anything free on Monday?")

    assert result.text == "I can do nine, half nine or ten."
    assert [record.tool_name for record in result.tool_calls] == [
        "check_availability"
    ]
    assert result.tool_calls[0].success is True


def test_the_tool_result_is_fed_back_to_the_model(
    dialogue, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(use_tools(_check()), say("Ten o'clock is free."))
    conversation = dialogue(model=model)

    conversation.send("Anything on Monday?")

    second_request = model.requests[1]["messages"]
    result_block = second_request[-1].content[0]
    assert result_block["type"] == "tool_result"
    assert result_block["is_error"] is False
    assert TEN in result_block["content"]


def test_check_then_book_produces_a_real_appointment(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(
        use_tools(_check()),
        use_tools(_book()),
        say("You're booked in for ten on Monday."),
    )

    result = conversation.send("Book me a haircut Monday morning.")

    assert result.booked_appointment_id is not None
    stored = session.execute(select(Appointment)).scalar_one()
    assert str(stored.id) == result.booked_appointment_id
    assert stored.call_id is not None


def test_several_tool_calls_in_one_response_return_together(
    dialogue, open_weekdays, haircut: Service
) -> None:
    """Splitting them teaches the model to stop asking in parallel."""
    model = FakeModel(
        use_tools(_check(), _check(day="2026-03-03")), say("Monday or Tuesday?")
    )
    conversation = dialogue(model=model)

    result = conversation.send("Monday or Tuesday?")

    assert len(result.tool_calls) == 2
    returned = model.requests[1]["messages"][-1]
    assert returned.role == "user"
    assert len(returned.content) == 2
    assert all(block["type"] == "tool_result" for block in returned.content)


def test_a_tool_failure_is_handed_back_for_the_model_to_recover_from(
    dialogue, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(
        use_tools(_check(service="Hovercraft")),
        say("I'm afraid we only do haircuts."),
    )
    conversation = dialogue(model=model)

    result = conversation.send("Can I book a hovercraft?")

    assert result.failed is False
    assert result.tool_calls[0].success is False
    block = model.requests[1]["messages"][-1].content[0]
    assert block["is_error"] is True
    assert json.loads(block["content"])["data"]["services_offered"] == ["Haircut"]


def test_a_successful_transfer_marks_the_turn_escalated(
    dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(
        use_tools(("transfer_to_human", {"reason": "Caller wants a refund."})),
        say("Let me put you through."),
    )

    result = conversation.send("I want my money back.")

    assert result.escalated is True


def test_a_failed_transfer_does_not_mark_the_turn_escalated(
    dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(
        use_tools(("transfer_to_human", {"reason": "   "})), say("One moment.")
    )

    result = conversation.send("Get me a person.")

    assert result.escalated is False


def test_a_blocked_booking_is_never_reported_as_booked(
    session: Session, dialogue, open_weekdays, haircut: Service
) -> None:
    """The guard runs inside the loop, so the result cannot claim otherwise."""
    conversation = dialogue(use_tools(_book()), say("I'll need to check that."))

    result = conversation.send("Book me in at ten.")

    assert result.booked_appointment_id is None
    assert result.tool_calls[0].success is False
    assert session.execute(select(Appointment)).first() is None


# --- failure ---------------------------------------------------------------


def test_an_unreachable_model_gives_a_safe_fixed_reply(
    dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(raises=ModelUnavailable("the api fell over"))

    result = conversation.send("Anything free on Monday?")

    assert result.text == MODEL_FAILURE_REPLY
    assert result.failed is True
    assert result.booked_appointment_id is None


def test_a_refusal_gives_the_same_safe_reply(
    dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(raises=ModelRefused("declined"))

    result = conversation.send("Tell me something unwise.")

    assert result.text == MODEL_FAILURE_REPLY
    assert result.failed is True


def test_a_model_failure_after_a_tool_ran_keeps_the_tool_record(
    dialogue, open_weekdays, haircut: Service
) -> None:
    """The audit trail is what the failure is for.

    The tool ran and the calendar answered; the model then fell over. That
    turn still has something worth reading afterwards.
    """

    class FailsOnSecondCall:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, *, system, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return use_tools(_check())
            raise ModelUnavailable("the api fell over")

    conversation = dialogue(model=FailsOnSecondCall())

    result = conversation.send("Anything Monday?")

    assert result.failed is True
    assert result.text == MODEL_FAILURE_REPLY
    assert [record.tool_name for record in result.tool_calls] == [
        "check_availability"
    ]
    assert result.tool_calls[0].success is True


def test_a_model_that_never_stops_asking_for_tools_is_cut_off(
    dialogue, calendar_settings, open_weekdays, haircut: Service
) -> None:
    settings = calendar_settings.model_copy(update={"max_tool_iterations": 3})
    model = FakeModel(*[use_tools(_check()) for _ in range(10)])
    conversation = dialogue(model=model, settings=settings)

    result = conversation.send("Anything Monday?")

    assert result.text == LOOP_EXHAUSTED_REPLY
    assert result.failed is True
    assert model.call_count == 3
    assert len(result.tool_calls) == 3


def test_the_loop_limit_comes_from_settings(
    dialogue, calendar_settings, open_weekdays, haircut: Service
) -> None:
    settings = calendar_settings.model_copy(update={"max_tool_iterations": 1})
    model = FakeModel(*[use_tools(_check()) for _ in range(5)])

    result = dialogue(model=model, settings=settings).send("Monday?")

    assert result.failed is True
    assert model.call_count == 1


def test_latency_is_summed_across_every_model_call_in_a_turn(
    dialogue, open_weekdays, haircut: Service
) -> None:
    conversation = dialogue(
        use_tools(_check(), latency_ms=40), say("Ten is free.", latency_ms=25)
    )

    result = conversation.send("Monday?")

    assert result.llm_latency_ms == 65


# --- what the model reported using (milestone 8) --------------------------


def test_a_turn_reports_the_tokens_the_model_said_it_used(
    dialogue, open_weekdays, haircut
) -> None:
    conversation = dialogue(used(1000, 200))

    result = conversation.send("Hello")

    assert result.input_tokens == 1000
    assert result.output_tokens == 200
    assert result.model_name == "fake-model-1"


def test_a_turn_that_used_several_requests_adds_them_up(
    dialogue, open_weekdays, haircut
) -> None:
    """One reply can be several requests, and every one of them is paid for."""
    asking = use_tools(
        ("check_availability", {"service_name": "Haircut", "date": "2026-03-02"})
    )
    conversation = dialogue(
        _with_usage(asking, 500, 50), used(700, 80, text="We have 9am free.")
    )

    result = conversation.send("Anything on Monday?")

    assert result.input_tokens == 1200
    assert result.output_tokens == 130


def test_a_model_that_reported_nothing_leaves_the_counts_null(
    dialogue, open_weekdays, haircut
) -> None:
    """Null is "we were not told", which is not "nothing was used"."""
    conversation = dialogue(say("Certainly."))

    result = conversation.send("Hello")

    assert result.input_tokens is None
    assert result.output_tokens is None


def test_a_zero_token_report_is_kept_as_zero(dialogue, open_weekdays, haircut) -> None:
    conversation = dialogue(used(0, 0))

    result = conversation.send("Hello")

    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_a_turn_whose_model_failed_still_reports_nothing_rather_than_zero(
    dialogue, open_weekdays, haircut
) -> None:
    from app.providers.llm import ModelError

    conversation = dialogue(raises=ModelError("down"))

    result = conversation.send("Hello")

    assert result.failed is True
    assert result.input_tokens is None


def test_a_turn_that_ran_out_of_iterations_still_reports_what_it_spent(
    dialogue, open_weekdays, haircut, calendar_settings
) -> None:
    """The requests it did make were paid for, loop or no loop."""
    settings = calendar_settings.model_copy(update={"max_tool_iterations": 2})
    asking = use_tools(
        ("check_availability", {"service_name": "Haircut", "date": "2026-03-02"})
    )
    conversation = dialogue(
        _with_usage(asking, 100, 10),
        _with_usage(asking, 100, 10),
        settings=settings,
    )

    result = conversation.send("Anything on Monday?")

    assert result.failed is True
    assert result.input_tokens == 200
    assert result.output_tokens == 20


def test_the_dialogue_layer_puts_no_price_on_any_of_it(
    dialogue, open_weekdays, haircut
) -> None:
    conversation = dialogue(used(1000, 200))

    result = conversation.send("Hello")

    assert not [field for field in vars(result) if "cost" in field or "usd" in field]


def _with_usage(response, input_tokens: int, output_tokens: int):
    """The same scripted response, with token usage attached."""
    import dataclasses

    return dataclasses.replace(
        response, input_tokens=input_tokens, output_tokens=output_tokens
    )
