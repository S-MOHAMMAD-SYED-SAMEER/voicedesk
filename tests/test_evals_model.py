"""The scripted model: determinism, and running out of script.

The second of those matters as much as the first. A scenario whose script is
too short must produce a **dataset** failure, clearly labelled, because
blaming the receptionist for a half-written scenario would make every number
in the report worthless.
"""

import pytest

from app.evals.model import (
    INPUT_TOKENS,
    MODEL_NAME,
    OUTPUT_TOKENS,
    ScriptedModel,
    ScriptExhausted,
    say,
    use_tools,
)
from app.providers.llm import ModelResponse

BOOK = ("book_appointment", {"service_name": "Haircut"})


def _ask(model: ScriptedModel) -> ModelResponse:
    return model.respond(system="prompt", messages=[], tools=[])


# --- determinism -----------------------------------------------------------


def test_responses_come_back_in_the_order_they_were_written() -> None:
    model = ScriptedModel([say("one"), say("two"), say("three")])

    assert [_ask(model).text for _ in range(3)] == ["one", "two", "three"]


def test_two_models_from_one_script_behave_identically() -> None:
    script = [use_tools(BOOK), say("Done.")]
    first, second = ScriptedModel(script), ScriptedModel(script)

    assert _ask(first).raw_content == _ask(second).raw_content
    assert _ask(first).text == _ask(second).text


def test_tool_identifiers_are_derived_rather_than_generated() -> None:
    """Two runs of one scenario must produce the same bytes."""
    first = use_tools(BOOK, ("cancel", {"appointment_id": "x"}))
    second = use_tools(BOOK, ("cancel", {"appointment_id": "x"}))

    assert [use.id for use in first.tool_uses] == [use.id for use in second.tool_uses]


def test_tool_identifiers_are_unique_within_one_response() -> None:
    response = use_tools(BOOK, BOOK)

    assert len({use.id for use in response.tool_uses}) == 2


# --- what it records -------------------------------------------------------


def test_the_call_count_starts_at_zero() -> None:
    assert ScriptedModel([say("one")]).call_count == 0


def test_the_call_count_counts_requests_not_responses() -> None:
    model = ScriptedModel([say("one"), say("two")])
    _ask(model)

    assert model.call_count == 1


def test_a_request_is_recorded_even_when_the_script_runs_out() -> None:
    """Silence must be provable: a model that was asked has a record of it."""
    model = ScriptedModel([])

    with pytest.raises(ScriptExhausted):
        _ask(model)

    assert model.call_count == 1


def test_what_the_model_was_asked_is_kept() -> None:
    model = ScriptedModel([say("one")])
    model.respond(system="the prompt", messages=["m"], tools=["t"])

    assert model.requests[0]["system"] == "the prompt"
    assert model.requests[0]["messages"] == ["m"]


def test_remaining_counts_down() -> None:
    model = ScriptedModel([say("one"), say("two")])
    assert model.remaining == 2
    _ask(model)
    assert model.remaining == 1


def test_a_model_with_no_script_is_exhausted_from_the_start() -> None:
    assert ScriptedModel([]).exhausted is True


# --- running out -----------------------------------------------------------


def test_running_out_raises_rather_than_inventing_a_reply() -> None:
    model = ScriptedModel([say("one")])
    _ask(model)

    with pytest.raises(ScriptExhausted):
        _ask(model)


def test_the_exhaustion_message_names_the_scenario() -> None:
    model = ScriptedModel([], scenario="booking_available")

    with pytest.raises(ScriptExhausted, match="booking_available"):
        _ask(model)


def test_the_exhaustion_message_explains_the_tool_loop() -> None:
    """The usual cause is forgetting that one turn can be several requests."""
    with pytest.raises(ScriptExhausted, match="tool loop"):
        _ask(ScriptedModel([]))


# --- the script helpers ----------------------------------------------------


def test_a_plain_reply_ends_the_turn() -> None:
    response = say("Of course.")

    assert response.stop_reason == "end_turn"
    assert response.tool_uses == []
    assert response.raw_content == [{"type": "text", "text": "Of course."}]


def test_a_tool_reply_carries_the_arguments_it_was_given() -> None:
    response = use_tools(BOOK)

    assert response.stop_reason == "tool_use"
    assert response.tool_uses[0].name == "book_appointment"
    assert response.tool_uses[0].arguments == {"service_name": "Haircut"}


def test_a_tool_reply_repeats_itself_in_raw_content() -> None:
    """The raw blocks are what gets replayed to the model, so they must agree."""
    response = use_tools(BOOK)
    block = response.raw_content[0]

    assert block["type"] == "tool_use"
    assert block["id"] == response.tool_uses[0].id
    assert block["input"] == response.tool_uses[0].arguments


def test_a_tool_reply_may_also_say_something() -> None:
    response = use_tools(BOOK, text="One moment.")

    assert response.text == "One moment."
    assert response.raw_content[0] == {"type": "text", "text": "One moment."}


def test_every_scripted_response_reports_fixed_usage() -> None:
    """Fixed, so two runs of one scenario record the same cost."""
    for response in (say("x"), use_tools(BOOK)):
        assert response.input_tokens == INPUT_TOKENS
        assert response.output_tokens == OUTPUT_TOKENS
        assert response.model_name == MODEL_NAME


def test_the_model_name_says_it_was_scripted() -> None:
    """Nothing reading a cost row should think a vendor was asked."""
    assert "script" in MODEL_NAME
