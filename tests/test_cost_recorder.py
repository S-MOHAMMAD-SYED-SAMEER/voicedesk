"""Writing cost rows, and totalling a call — against a real database.

Three properties are load-bearing and each has its own section:

* usage is written whether or not it can be priced;
* one unpriced component leaves the whole call's total null;
* recording the same turn twice records it once.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.config import Settings
from app.cost.recorder import record_call_cost, record_turn_cost, recompute_total
from app.models import Call, CallCost, CostComponent, Turn, TurnRole
from tests.conftest import cost_rows


@dataclass
class Dialogue:
    """Just enough of a `DialogueResult` for the recorder."""

    caller_turn_id: uuid.UUID | None
    agent_turn_id: uuid.UUID | None
    model_name: str = "some-model"
    input_tokens: int | None = 1000
    output_tokens: int | None = 200


@dataclass
class Speech:
    """Just enough of a turn's speech usage for the recorder."""

    stt_provider: str = "deepgram"
    stt_audio_ms: int | None = 3000
    tts_provider: str = "elevenlabs"
    tts_characters: int | None = 100


@pytest.fixture
def turns(session, call) -> tuple[Turn, Turn]:
    """One caller turn and one agent turn, as the dialogue layer writes them."""
    caller = Turn(call_id=call.id, role=TurnRole.CALLER, text="Hello")
    agent = Turn(call_id=call.id, role=TurnRole.AGENT, text="Hello there")
    session.add_all([caller, agent])
    session.commit()
    return caller, agent


@pytest.fixture
def dialogue(turns) -> Dialogue:
    caller, agent = turns
    return Dialogue(caller_turn_id=caller.id, agent_turn_id=agent.id)


# --- the switch -----------------------------------------------------------


def test_nothing_is_written_when_cost_tracking_is_off(
    session, call, dialogue, calendar_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), calendar_settings, llm_provider="anthropic"
    )

    assert cost_rows(session, call.id) == {}


def test_the_total_stays_null_when_cost_tracking_is_off(
    session, call, dialogue, calendar_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), calendar_settings, llm_provider="anthropic"
    )
    session.refresh(call)

    assert call.total_cost_usd is None


# --- what one turn writes -------------------------------------------------


def test_a_turn_writes_one_row_per_component(
    session, call, dialogue, cost_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )

    assert set(cost_rows(session, call.id)) == {
        CostComponent.LLM,
        CostComponent.STT,
        CostComponent.TTS,
    }


def test_the_model_row_records_both_token_counts(
    session, call, dialogue, cost_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )
    row = cost_rows(session, call.id)[CostComponent.LLM]

    assert row.input_units == Decimal("1000.000")
    assert row.output_units == Decimal("200.000")
    assert row.unit_type == "tokens"
    assert row.provider == "anthropic"
    assert row.model == "some-model"


def test_recognition_is_recorded_against_the_callers_own_turn(
    session, call, dialogue, turns, cost_settings: Settings
) -> None:
    """Where `turns.audio_ms` already lives, for the same reason."""
    caller, _agent = turns
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )

    assert cost_rows(session, call.id)[CostComponent.STT].turn_id == caller.id


def test_the_model_and_synthesis_are_recorded_against_the_reply(
    session, call, dialogue, turns, cost_settings: Settings
) -> None:
    _caller, agent = turns
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )
    rows = cost_rows(session, call.id)

    assert rows[CostComponent.LLM].turn_id == agent.id
    assert rows[CostComponent.TTS].turn_id == agent.id


def test_a_component_with_no_usage_writes_no_row(
    session, call, dialogue, cost_settings: Settings
) -> None:
    """Silence from a provider is not a free turn; it is no row at all."""
    record_turn_cost(
        session,
        dialogue,
        Speech(stt_audio_ms=None),
        cost_settings,
        llm_provider="anthropic",
    )

    assert CostComponent.STT not in cost_rows(session, call.id)


def test_a_model_that_reported_no_tokens_writes_no_model_row(
    session, call, turns, cost_settings: Settings
) -> None:
    caller, agent = turns
    result = Dialogue(caller.id, agent.id, input_tokens=None, output_tokens=None)

    record_turn_cost(session, result, Speech(), cost_settings, llm_provider="anthropic")

    assert CostComponent.LLM not in cost_rows(session, call.id)


def test_no_model_row_is_written_without_a_provider_to_attribute_it_to(
    session, call, dialogue, cost_settings: Settings
) -> None:
    record_turn_cost(session, dialogue, Speech(), cost_settings, llm_provider="")

    assert CostComponent.LLM not in cost_rows(session, call.id)


def test_a_turn_with_no_agent_row_writes_nothing(
    session, call, turns, cost_settings: Settings
) -> None:
    caller, _agent = turns
    result = Dialogue(caller_turn_id=caller.id, agent_turn_id=None)

    record_turn_cost(session, result, Speech(), cost_settings, llm_provider="anthropic")

    assert cost_rows(session, call.id) == {}


def test_a_turn_that_no_call_owns_writes_nothing_and_does_not_raise(
    session, call, cost_settings: Settings
) -> None:
    result = Dialogue(caller_turn_id=uuid.uuid4(), agent_turn_id=uuid.uuid4())

    record_turn_cost(session, result, Speech(), cost_settings, llm_provider="anthropic")

    assert cost_rows(session, call.id) == {}


# --- priced and unpriced --------------------------------------------------


def test_usage_is_kept_even_when_nothing_can_be_priced(
    session, call, dialogue, cost_settings: Settings
) -> None:
    """The measurement is a fact; the price is the part nobody supplied."""
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )
    row = cost_rows(session, call.id)[CostComponent.STT]

    assert row.input_units == Decimal("3000.000")
    assert row.cost_usd is None
    assert row.unit_price_usd is None


def test_an_unpriced_row_says_why(
    session, call, dialogue, cost_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )
    row = cost_rows(session, call.id)[CostComponent.TTS]

    assert row.details["unpriced_reason"] == "no_price_configured"


def test_a_configured_price_produces_a_cost(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    rows = cost_rows(session, call.id)

    # 1000 input at 0.000001 plus 200 output at 0.000004.
    assert rows[CostComponent.LLM].cost_usd == Decimal("0.001800")
    # 3000 ms at 0.000002 per ms.
    assert rows[CostComponent.STT].cost_usd == Decimal("0.006000")
    # 100 characters at 0.0005 each.
    assert rows[CostComponent.TTS].cost_usd == Decimal("0.050000")


def test_a_single_rate_row_stores_the_rate_it_used(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    row = cost_rows(session, call.id)[CostComponent.STT]

    assert row.unit_price_usd is not None
    assert row.cost_usd == (row.input_units * row.unit_price_usd).quantize(
        Decimal("0.000001")
    )


def test_a_model_row_stores_both_rates_beside_the_figure(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    row = cost_rows(session, call.id)[CostComponent.LLM]

    assert row.unit_price_usd is None
    assert Decimal(row.details["input_usd_per_unit"]) == Decimal("0.000001")
    assert Decimal(row.details["output_usd_per_unit"]) == Decimal("0.000004")


def test_an_offline_provider_costs_zero_rather_than_nothing(
    session, call, dialogue, cost_settings: Settings
) -> None:
    record_turn_cost(
        session,
        dialogue,
        Speech(stt_provider="offline", tts_provider="offline"),
        cost_settings,
        llm_provider="anthropic",
    )
    rows = cost_rows(session, call.id)

    assert rows[CostComponent.STT].cost_usd == Decimal("0.000000")
    assert rows[CostComponent.TTS].cost_usd == Decimal("0.000000")


def test_a_cost_comes_back_as_a_decimal_not_a_float(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    row = cost_rows(session, call.id)[CostComponent.TTS]

    assert isinstance(row.cost_usd, Decimal)


# --- the call's total -----------------------------------------------------


def test_the_total_is_null_while_any_component_is_unpriced(
    session, call, dialogue, cost_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )
    session.refresh(call)

    assert call.total_cost_usd is None


def test_one_unpriced_component_among_priced_ones_still_nulls_the_total(
    session, call, dialogue, priced_settings: Settings
) -> None:
    """A partial total looks exactly like a complete one, so there is none."""
    settings = priced_settings.model_copy(update={"stt_usd_per_minute": ""})
    record_turn_cost(session, dialogue, Speech(), settings, llm_provider="anthropic")
    session.refresh(call)

    rows = cost_rows(session, call.id)
    assert rows[CostComponent.LLM].cost_usd is not None
    assert rows[CostComponent.STT].cost_usd is None
    assert call.total_cost_usd is None


def test_the_total_is_the_sum_when_everything_is_priced(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    session.refresh(call)

    assert call.total_cost_usd == Decimal("0.057800")


def test_a_call_with_no_cost_rows_has_no_total(session, call) -> None:
    assert recompute_total(session, call.id) is None


def test_the_total_is_recomputed_from_the_rows_not_added_to(
    session, call, dialogue, priced_settings: Settings
) -> None:
    """Read-modify-write on a column two turns can finish at once drifts."""
    call.total_cost_usd = Decimal("999.000000")
    session.commit()

    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    session.refresh(call)

    assert call.total_cost_usd == Decimal("0.057800")


def test_recomputing_twice_gives_the_same_total(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )

    assert recompute_total(session, call.id) == recompute_total(session, call.id)


def test_a_total_too_large_to_store_is_left_null(
    session, call, dialogue, cost_settings: Settings
) -> None:
    session.add(
        CallCost(
            call_id=call.id,
            component=CostComponent.TTS,
            provider="elevenlabs",
            input_units=Decimal("1"),
            unit_type="characters",
            cost_usd=Decimal("9999.999999"),
        )
    )
    session.add(
        CallCost(
            call_id=call.id,
            turn_id=dialogue.agent_turn_id,
            component=CostComponent.LLM,
            provider="anthropic",
            input_units=Decimal("1"),
            unit_type="tokens",
            cost_usd=Decimal("1.000000"),
        )
    )
    session.commit()

    assert recompute_total(session, call.id) is None


def test_two_turns_both_count_towards_the_total(
    session, call, turns, priced_settings: Settings
) -> None:
    caller, agent = turns
    second_caller = Turn(call_id=call.id, role=TurnRole.CALLER, text="And again")
    second_agent = Turn(call_id=call.id, role=TurnRole.AGENT, text="Of course")
    session.add_all([second_caller, second_agent])
    session.commit()

    record_turn_cost(
        session,
        Dialogue(caller.id, agent.id),
        Speech(),
        priced_settings,
        llm_provider="anthropic",
    )
    record_turn_cost(
        session,
        Dialogue(second_caller.id, second_agent.id),
        Speech(),
        priced_settings,
        llm_provider="anthropic",
    )
    session.refresh(call)

    assert len(session.execute(select(CallCost)).scalars().all()) == 6
    assert call.total_cost_usd == Decimal("0.115600")


# --- recording twice ------------------------------------------------------


def test_recording_the_same_turn_twice_writes_three_rows_not_six(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )

    assert len(session.execute(select(CallCost)).scalars().all()) == 3


def test_recording_twice_does_not_double_the_total(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    session.refresh(call)

    assert call.total_cost_usd == Decimal("0.057800")


def test_a_second_recording_corrects_the_first(
    session, call, dialogue, priced_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    record_turn_cost(
        session,
        dialogue,
        Speech(tts_characters=200),
        priced_settings,
        llm_provider="anthropic",
    )
    row = cost_rows(session, call.id)[CostComponent.TTS]

    assert row.input_units == Decimal("200.000")
    assert row.cost_usd == Decimal("0.100000")


def test_the_database_refuses_a_second_row_for_one_turn_and_component(
    session, call, turns
) -> None:
    """Idempotency is a constraint, not only an application habit."""
    import sqlalchemy.exc

    _caller, agent = turns
    for _ in range(2):
        session.add(
            CallCost(
                call_id=call.id,
                turn_id=agent.id,
                component=CostComponent.LLM,
                provider="anthropic",
                input_units=Decimal("1"),
                unit_type="tokens",
            )
        )
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        session.commit()
    session.rollback()


# --- the line -------------------------------------------------------------


def test_a_finished_call_records_how_long_the_line_was_open(
    session, call, cost_settings: Settings
) -> None:
    call.started_at = datetime.now(UTC) - timedelta(seconds=61)
    call.ended_at = datetime.now(UTC)
    session.commit()

    record_call_cost(session, call, "twilio", cost_settings)
    row = cost_rows(session, call.id)[CostComponent.TELEPHONY]

    assert row.unit_type == "duration_ms"
    assert row.input_units >= Decimal("61000")
    assert row.turn_id is None


def test_the_line_is_recorded_unpriced_however_the_prices_are_set(
    session, call, priced_settings: Settings
) -> None:
    """Measured stream time is not what a carrier bills for."""
    call.started_at = datetime.now(UTC) - timedelta(seconds=61)
    call.ended_at = datetime.now(UTC)
    session.commit()

    record_call_cost(session, call, "twilio", priced_settings)
    row = cost_rows(session, call.id)[CostComponent.TELEPHONY]

    assert row.cost_usd is None
    assert row.unit_price_usd is None


def test_a_telephone_call_therefore_has_no_total(
    session, call, dialogue, priced_settings: Settings
) -> None:
    """The honest consequence of not pricing the line, asserted rather than hidden."""
    call.started_at = datetime.now(UTC) - timedelta(seconds=61)
    call.ended_at = datetime.now(UTC)
    session.commit()

    record_turn_cost(
        session, dialogue, Speech(), priced_settings, llm_provider="anthropic"
    )
    record_call_cost(session, call, "twilio", priced_settings)
    session.refresh(call)

    assert call.total_cost_usd is None


def test_a_call_still_in_progress_records_no_line_usage(
    session, call, cost_settings: Settings
) -> None:
    assert call.ended_at is None

    record_call_cost(session, call, "twilio", cost_settings)

    assert cost_rows(session, call.id) == {}


def test_recording_the_line_twice_writes_one_row(
    session, call, cost_settings: Settings
) -> None:
    """The `(turn_id, component)` index cannot see two nulls as equal."""
    call.started_at = datetime.now(UTC) - timedelta(seconds=30)
    call.ended_at = datetime.now(UTC)
    session.commit()

    record_call_cost(session, call, "twilio", cost_settings)
    record_call_cost(session, call, "twilio", cost_settings)

    assert len(session.execute(select(CallCost)).scalars().all()) == 1


def test_the_database_refuses_a_second_line_row_for_one_call(session, call) -> None:
    """The partial index, which is what makes the row above idempotent."""
    import sqlalchemy.exc

    for _ in range(2):
        session.add(
            CallCost(
                call_id=call.id,
                component=CostComponent.TELEPHONY,
                provider="twilio",
                input_units=Decimal("1000"),
                unit_type="duration_ms",
            )
        )
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        session.commit()
    session.rollback()


def test_two_different_calls_each_get_their_own_line_row(
    session, call, cost_settings: Settings
) -> None:
    other = Call(from_number="+447700900999", to_number="+441234567890")
    session.add(other)
    session.commit()

    for row in (call, other):
        row.started_at = datetime.now(UTC) - timedelta(seconds=10)
        row.ended_at = datetime.now(UTC)
    session.commit()

    record_call_cost(session, call, "twilio", cost_settings)
    record_call_cost(session, other, "twilio", cost_settings)

    assert len(session.execute(select(CallCost)).scalars().all()) == 2


def test_a_call_that_ended_before_it_started_records_nothing(
    session, call, cost_settings: Settings
) -> None:
    call.started_at = datetime.now(UTC)
    call.ended_at = datetime.now(UTC) - timedelta(seconds=5)
    session.commit()

    record_call_cost(session, call, "twilio", cost_settings)

    assert cost_rows(session, call.id) == {}


def test_no_line_row_is_written_without_a_carrier_to_attribute_it_to(
    session, call, cost_settings: Settings
) -> None:
    call.started_at = datetime.now(UTC) - timedelta(seconds=10)
    call.ended_at = datetime.now(UTC)
    session.commit()

    record_call_cost(session, call, "", cost_settings)

    assert cost_rows(session, call.id) == {}


# --- what happens to rows when rows go away -------------------------------


def test_deleting_a_call_deletes_its_cost_rows(
    session, call, dialogue, cost_settings: Settings
) -> None:
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )

    session.delete(call)
    session.commit()

    assert session.execute(select(CallCost)).scalars().all() == []


def test_deleting_a_turn_keeps_what_it_spent(
    session, call, dialogue, turns, cost_settings: Settings
) -> None:
    """The call still owns it. Money spent does not become unspent."""
    _caller, agent = turns
    record_turn_cost(
        session, dialogue, Speech(), cost_settings, llm_provider="anthropic"
    )

    session.delete(agent)
    session.commit()

    rows = cost_rows(session, call.id)
    assert rows[CostComponent.LLM].turn_id is None
    assert rows[CostComponent.LLM].call_id == call.id
