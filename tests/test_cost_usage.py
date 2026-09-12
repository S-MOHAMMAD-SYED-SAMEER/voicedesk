"""What was consumed, measured — and what happens when it was not measured.

The distinction this file exists to protect: **unknown is not zero.** A
provider that reported nothing produces no usage record at all, so nothing
downstream can mistake its silence for a free turn.
"""

from decimal import Decimal

import pytest

from app.cost.usage import (
    AUDIO_MS,
    CHARACTERS,
    DURATION_MS,
    TOKENS,
    UNIT_FOR_COMPONENT,
    UNIT_TYPES,
    Usage,
    UsageError,
    llm_usage,
    stt_usage,
    telephony_usage,
    tts_usage,
)
from app.models import CostComponent


# --- the record itself ----------------------------------------------------


def test_a_usage_record_names_its_unit_from_its_component() -> None:
    usage = Usage(
        component=CostComponent.STT, provider="offline", input_units=Decimal("20")
    )

    assert usage.unit_type == AUDIO_MS


def test_every_component_has_exactly_one_unit() -> None:
    """A component measured two ways could not be totalled."""
    assert set(UNIT_FOR_COMPONENT) == set(CostComponent)
    assert set(UNIT_FOR_COMPONENT.values()) <= set(UNIT_TYPES)


def test_the_four_units_are_distinct() -> None:
    assert len(set(UNIT_TYPES)) == 4
    assert set(UNIT_TYPES) == {TOKENS, AUDIO_MS, CHARACTERS, DURATION_MS}


def test_total_units_adds_input_and_output() -> None:
    usage = Usage(
        component=CostComponent.LLM,
        provider="anthropic",
        input_units=Decimal("100"),
        output_units=Decimal("25"),
    )

    assert usage.total_units == Decimal("125")


def test_output_units_default_to_zero() -> None:
    usage = Usage(
        component=CostComponent.TTS, provider="offline", input_units=Decimal("9")
    )

    assert usage.output_units == Decimal("0")


def test_negative_input_units_are_refused() -> None:
    with pytest.raises(UsageError, match="cannot be negative"):
        Usage(
            component=CostComponent.STT,
            provider="offline",
            input_units=Decimal("-1"),
        )


def test_negative_output_units_are_refused() -> None:
    with pytest.raises(UsageError, match="cannot be negative"):
        Usage(
            component=CostComponent.LLM,
            provider="anthropic",
            input_units=Decimal("1"),
            output_units=Decimal("-1"),
        )


def test_float_units_are_refused() -> None:
    """Money is never computed from a float, so units are never floats."""
    with pytest.raises(UsageError, match="Decimal"):
        Usage(
            component=CostComponent.STT,
            provider="offline",
            input_units=1500.0,  # type: ignore[arg-type]
        )


def test_an_integer_is_not_a_decimal_either() -> None:
    with pytest.raises(UsageError, match="Decimal"):
        Usage(
            component=CostComponent.TTS,
            provider="offline",
            input_units=12,  # type: ignore[arg-type]
        )


def test_a_usage_record_must_name_who_did_the_work() -> None:
    with pytest.raises(UsageError, match="who did the work"):
        Usage(
            component=CostComponent.STT, provider="", input_units=Decimal("1")
        )


def test_zero_units_are_a_measurement_and_are_allowed() -> None:
    """Nothing consumed is a fact. Nothing *reported* is the other case."""
    usage = Usage(
        component=CostComponent.TTS, provider="offline", input_units=Decimal("0")
    )

    assert usage.input_units == Decimal("0")


def test_a_usage_record_cannot_be_mutated() -> None:
    usage = Usage(
        component=CostComponent.STT, provider="offline", input_units=Decimal("1")
    )

    with pytest.raises(Exception):
        usage.input_units = Decimal("2")  # type: ignore[misc]


# --- the model ------------------------------------------------------------


def test_model_usage_records_both_token_counts() -> None:
    usage = llm_usage("anthropic", "some-model", 120, 34)

    assert usage is not None
    assert usage.component is CostComponent.LLM
    assert usage.input_units == Decimal("120")
    assert usage.output_units == Decimal("34")
    assert usage.unit_type == TOKENS
    assert usage.model == "some-model"


def test_model_usage_is_nothing_when_input_tokens_were_not_reported() -> None:
    assert llm_usage("anthropic", "some-model", None, 34) is None


def test_model_usage_is_nothing_when_output_tokens_were_not_reported() -> None:
    """Half a measurement priced as a whole one is wrong invisibly."""
    assert llm_usage("anthropic", "some-model", 120, None) is None


def test_model_usage_is_nothing_when_neither_was_reported() -> None:
    assert llm_usage("anthropic", "some-model", None, None) is None


def test_model_usage_says_plainly_that_cache_tokens_are_not_measured() -> None:
    """A limitation recorded on the row, not hidden in a commit message."""
    usage = llm_usage("anthropic", "some-model", 1, 1)

    assert usage is not None
    assert usage.metadata["cache_tokens_measured"] is False


def test_model_usage_keeps_a_zero_token_count() -> None:
    usage = llm_usage("anthropic", "some-model", 0, 0)

    assert usage is not None
    assert usage.input_units == Decimal("0")


def test_model_usage_without_a_model_name_is_still_recorded() -> None:
    usage = llm_usage("anthropic", None, 5, 5)

    assert usage is not None
    assert usage.model is None


# --- recognition ----------------------------------------------------------


def test_recognition_usage_records_the_audio_it_was_given() -> None:
    usage = stt_usage("deepgram", 1500)

    assert usage is not None
    assert usage.component is CostComponent.STT
    assert usage.input_units == Decimal("1500")
    assert usage.unit_type == AUDIO_MS
    assert usage.output_units == Decimal("0")


def test_recognition_usage_is_nothing_when_no_duration_was_reported() -> None:
    assert stt_usage("deepgram", None) is None


def test_recognition_usage_says_what_it_actually_measured() -> None:
    """Audio inside the results, which is less than connection time."""
    usage = stt_usage("deepgram", 1500)

    assert usage is not None
    assert usage.metadata["measured"] == "audio_in_results"


# --- synthesis ------------------------------------------------------------


def test_synthesis_usage_records_the_characters_it_was_given() -> None:
    usage = tts_usage("elevenlabs", 100)

    assert usage is not None
    assert usage.component is CostComponent.TTS
    assert usage.input_units == Decimal("100")
    assert usage.unit_type == CHARACTERS


def test_synthesis_usage_is_nothing_when_no_count_was_reported() -> None:
    assert tts_usage("elevenlabs", None) is None


def test_synthesis_usage_takes_one_number_not_a_stream() -> None:
    """The builder cannot be handed chunks, so it cannot sum them.

    A hundred characters delivered in ten chunks is a hundred characters. The
    only way to record a thousand would be to add the chunks up, and there is
    nowhere here to do that.
    """
    usage = tts_usage("elevenlabs", 100)

    assert usage is not None
    assert usage.input_units == Decimal("100")
    assert usage.metadata["counted"] == "once_per_stream"


# --- the line -------------------------------------------------------------


def test_line_usage_records_the_measured_duration() -> None:
    usage = telephony_usage("twilio", 61_000)

    assert usage is not None
    assert usage.component is CostComponent.TELEPHONY
    assert usage.input_units == Decimal("61000")
    assert usage.unit_type == DURATION_MS


def test_line_usage_is_nothing_when_the_call_has_no_duration() -> None:
    assert telephony_usage("twilio", None) is None


def test_line_usage_says_that_it_is_wall_clock_and_not_billed_time() -> None:
    usage = telephony_usage("twilio", 61_000)

    assert usage is not None
    assert "wall_clock" in usage.metadata["measured"]


# --- the module as a whole ------------------------------------------------


def test_this_module_imports_nothing_that_knows_a_price() -> None:
    """Pricing lives in exactly one module, and this is not it.

    Checked against the code rather than the prose: the docstrings here point
    at `pricing.py`, which is the opposite of a problem.
    """
    import ast
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent / "app" / "cost" / "usage.py"
    ).read_text()
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "app.cost.pricing" not in imports
    assert imports == {"dataclasses", "decimal", "typing", "app.models"}


def test_no_name_defined_here_is_about_money() -> None:
    """No rate, no price, no total in dollars — only counts of things."""
    import ast
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent / "app" / "cost" / "usage.py"
    ).read_text()
    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }

    for name in defined:
        lowered = name.lower()
        assert "price" not in lowered, name
        assert "usd" not in lowered, name
        assert "cost" not in lowered or name == "UsageError", name
