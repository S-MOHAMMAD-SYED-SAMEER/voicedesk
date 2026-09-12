"""The only module that knows a price, and the things it must never do.

Chief among them: **VoiceDesk ships no vendor prices.** The first test in this
file is the one that matters. The rest are about arithmetic and about failing
loudly when an operator's configuration is wrong.

Every number below is invented. None of them is anybody's price.
"""

import ast
import pathlib
from decimal import Decimal

import pytest

from app.config import Settings
from app.cost.pricing import (
    BUILT_IN_RATES,
    MAX_COST_USD,
    MAX_USD_PER_MCHAR,
    MAX_USD_PER_MINUTE,
    MAX_USD_PER_MTOK,
    OFFLINE_PROVIDER,
    PriceBook,
    PricingError,
    Rates,
    parse_price,
    price,
)
from app.cost.usage import Usage, llm_usage, stt_usage, telephony_usage, tts_usage
from app.models import CostComponent

PRICING = pathlib.Path(__file__).resolve().parent.parent / "app" / "cost" / "pricing.py"
APP = pathlib.Path(__file__).resolve().parent.parent / "app"

VENDORS = ("anthropic", "deepgram", "elevenlabs", "twilio", "openai", "google")


# --- the promise ----------------------------------------------------------


def test_the_shipped_table_contains_no_vendor() -> None:
    """The whole of milestone 8's pricing policy, as one assertion."""
    assert {provider for _, provider in BUILT_IN_RATES} == {OFFLINE_PROVIDER}


def test_every_shipped_rate_is_zero() -> None:
    """They are free because nothing is bought, not because we guessed."""
    for rates in BUILT_IN_RATES.values():
        assert rates.input_usd_per_unit == Decimal("0")
        assert rates.output_usd_per_unit == Decimal("0")


def test_the_shipped_table_prices_no_model_and_no_line() -> None:
    """There is no offline model and no offline carrier to be free."""
    components = {component for component, _ in BUILT_IN_RATES}

    assert components == {CostComponent.STT, CostComponent.TTS}


def test_no_vendor_is_named_in_the_pricing_code() -> None:
    """Checked against the code, not the prose.

    The module docstring names all four vendors on purpose — it is the
    paragraph explaining that none of their prices are here. What must not
    exist is a vendor name the code can reach.
    """
    tree = ast.parse(PRICING.read_text())
    for node in ast.walk(tree):
        # Docstrings are the first statement of a module, class or function.
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ):
                body[0].value.value = ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for vendor in VENDORS:
                assert vendor not in node.value.lower(), node.value
        if isinstance(node, ast.Name):
            for vendor in VENDORS:
                assert vendor not in node.id.lower(), node.id


def test_a_fresh_configuration_prices_nothing() -> None:
    book = PriceBook.from_settings(Settings(_env_file=None))

    assert book.llm is None
    assert book.stt is None
    assert book.tts is None


def test_an_unconfigured_vendor_component_has_no_rate() -> None:
    book = PriceBook.from_settings(Settings(_env_file=None))
    usage = llm_usage("anthropic", "some-model", 100, 10)

    assert usage is not None
    assert book.rates_for(usage) is None


# --- reading a configured price -------------------------------------------


def test_an_empty_price_means_unpriced_not_free() -> None:
    assert parse_price("", setting="X", maximum=MAX_USD_PER_MTOK) is None


def test_whitespace_is_also_empty() -> None:
    assert parse_price("   ", setting="X", maximum=MAX_USD_PER_MTOK) is None


def test_an_explicit_zero_is_a_price() -> None:
    """An operator who says zero has said something; blank has not."""
    assert parse_price("0", setting="X", maximum=MAX_USD_PER_MTOK) == Decimal("0")


def test_a_plain_decimal_is_read_exactly() -> None:
    assert parse_price("1.25", setting="X", maximum=MAX_USD_PER_MTOK) == Decimal(
        "1.25"
    )


def test_a_price_that_is_not_a_number_fails_by_name() -> None:
    with pytest.raises(PricingError, match="VOICEDESK_FAKE"):
        parse_price("about three", setting="VOICEDESK_FAKE", maximum=MAX_USD_PER_MTOK)


def test_a_price_with_a_currency_symbol_fails() -> None:
    with pytest.raises(PricingError, match="not a number"):
        parse_price("$3.00", setting="X", maximum=MAX_USD_PER_MTOK)


def test_a_negative_price_fails() -> None:
    with pytest.raises(PricingError, match="cannot be negative"):
        parse_price("-1", setting="X", maximum=MAX_USD_PER_MTOK)


def test_an_infinite_price_fails() -> None:
    with pytest.raises(PricingError, match="finite"):
        parse_price("Infinity", setting="X", maximum=MAX_USD_PER_MTOK)


def test_a_price_that_is_not_a_number_at_all_fails() -> None:
    with pytest.raises(PricingError, match="finite"):
        parse_price("NaN", setting="X", maximum=MAX_USD_PER_MTOK)


def test_a_price_above_the_sanity_limit_fails() -> None:
    """A misplaced decimal point should not quietly bill five figures."""
    with pytest.raises(PricingError, match="sanity limit"):
        parse_price("99999", setting="X", maximum=MAX_USD_PER_MTOK)


def test_the_sanity_limits_are_absurd_rather_than_plausible() -> None:
    """They are guards, not knowledge of anybody's price list."""
    assert MAX_USD_PER_MTOK == Decimal("10000")
    assert MAX_USD_PER_MINUTE == Decimal("100")
    assert MAX_USD_PER_MCHAR == Decimal("10000")


def test_a_price_exactly_at_the_limit_is_accepted() -> None:
    assert parse_price(
        str(MAX_USD_PER_MTOK), setting="X", maximum=MAX_USD_PER_MTOK
    ) == MAX_USD_PER_MTOK


# --- converting to per-unit rates -----------------------------------------


def test_a_price_per_million_tokens_becomes_a_price_per_token(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)

    assert book.llm is not None
    # 1 USD per million tokens is one millionth of a dollar per token.
    assert book.llm.input_usd_per_unit == Decimal("0.000001")
    assert book.llm.output_usd_per_unit == Decimal("0.000004")


def test_a_price_per_minute_becomes_a_price_per_millisecond(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)

    assert book.stt is not None
    # 0.12 USD per minute over 60000 ms.
    assert book.stt.input_usd_per_unit == Decimal("0.000002")


def test_a_price_per_million_characters_becomes_a_price_per_character(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)

    assert book.tts is not None
    assert book.tts.input_usd_per_unit == Decimal("0.0005")


def test_half_a_model_price_is_refused(cost_settings: Settings) -> None:
    """Charging for input and not output would under-state, invisibly."""
    settings = cost_settings.model_copy(update={"llm_input_usd_per_mtok": "1"})

    with pytest.raises(PricingError, match="or neither"):
        PriceBook.from_settings(settings)


def test_the_other_half_is_refused_too(cost_settings: Settings) -> None:
    settings = cost_settings.model_copy(update={"llm_output_usd_per_mtok": "4"})

    with pytest.raises(PricingError, match="or neither"):
        PriceBook.from_settings(settings)


def test_a_malformed_price_names_the_environment_variable(
    cost_settings: Settings,
) -> None:
    settings = cost_settings.model_copy(update={"stt_usd_per_minute": "cheap"})

    with pytest.raises(PricingError, match="VOICEDESK_STT_USD_PER_MINUTE"):
        PriceBook.from_settings(settings)


def test_one_component_can_be_priced_while_others_are_not(
    cost_settings: Settings,
) -> None:
    settings = cost_settings.model_copy(update={"tts_usd_per_mchar": "500"})
    book = PriceBook.from_settings(settings)

    assert book.tts is not None
    assert book.llm is None
    assert book.stt is None


# --- which rate applies ---------------------------------------------------


def test_an_offline_provider_is_free_even_when_a_price_is_configured(
    priced_settings: Settings,
) -> None:
    """This repository knows its own code buys nothing; config cannot argue."""
    book = PriceBook.from_settings(priced_settings)
    usage = tts_usage(OFFLINE_PROVIDER, 100)

    assert usage is not None
    rates = book.rates_for(usage)
    assert rates is not None
    assert rates.input_usd_per_unit == Decimal("0")


def test_a_vendor_provider_takes_the_configured_rate(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = tts_usage("elevenlabs", 100)

    assert usage is not None
    rates = book.rates_for(usage)
    assert rates is not None
    assert rates.input_usd_per_unit == Decimal("0.0005")


def test_the_line_is_never_priced_however_it_is_configured(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = telephony_usage("twilio", 60_000)

    assert usage is not None
    assert book.rates_for(usage) is None


def test_the_line_is_not_priced_even_for_an_offline_provider(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = telephony_usage(OFFLINE_PROVIDER, 60_000)

    assert usage is not None
    assert book.rates_for(usage) is None


# --- costing --------------------------------------------------------------


def test_an_unpriced_component_costs_nothing_known(cost_settings: Settings) -> None:
    book = PriceBook.from_settings(cost_settings)
    usage = llm_usage("anthropic", "some-model", 1000, 100)

    assert usage is not None
    costed = price(usage, book)
    assert costed.cost_usd is None
    assert costed.unit_price_usd is None
    assert costed.priced is False
    assert costed.detail["unpriced_reason"] == "no_price_configured"


def test_a_model_turn_costs_input_plus_output(priced_settings: Settings) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = llm_usage("anthropic", "some-model", 1_000_000, 1_000_000)

    assert usage is not None
    costed = price(usage, book)
    assert costed.cost_usd == Decimal("5.000000")


def test_a_model_turn_leaves_the_unit_price_column_empty(
    priced_settings: Settings,
) -> None:
    """One column cannot honestly hold two rates."""
    book = PriceBook.from_settings(priced_settings)
    usage = llm_usage("anthropic", "some-model", 100, 100)

    assert usage is not None
    costed = price(usage, book)
    assert costed.unit_price_usd is None
    assert Decimal(costed.detail["input_usd_per_unit"]) == Decimal("0.000001")
    assert Decimal(costed.detail["output_usd_per_unit"]) == Decimal("0.000004")


def test_a_single_rate_component_records_the_rate_it_used(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = tts_usage("elevenlabs", 1000)

    assert usage is not None
    costed = price(usage, book)
    assert costed.unit_price_usd == Decimal("0.0005")
    assert costed.cost_usd == Decimal("0.500000")


def test_cost_equals_the_stored_rate_times_the_units(
    priced_settings: Settings,
) -> None:
    """What the column arithmetic promises, checked."""
    book = PriceBook.from_settings(priced_settings)
    usage = stt_usage("deepgram", 90_000)

    assert usage is not None
    costed = price(usage, book)
    assert costed.unit_price_usd is not None
    assert costed.cost_usd == (
        usage.input_units * costed.unit_price_usd
    ).quantize(Decimal("0.000001"))


def test_an_offline_component_costs_exactly_zero_and_not_nothing(
    priced_settings: Settings,
) -> None:
    """Zero and null are different answers, and this one is zero."""
    book = PriceBook.from_settings(priced_settings)
    usage = stt_usage(OFFLINE_PROVIDER, 5000)

    assert usage is not None
    costed = price(usage, book)
    assert costed.cost_usd == Decimal("0.000000")
    assert costed.priced is True


def test_a_cost_is_quantised_to_six_decimal_places(
    priced_settings: Settings,
) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = stt_usage("deepgram", 1)

    assert usage is not None
    costed = price(usage, book)
    assert costed.cost_usd is not None
    assert costed.cost_usd.as_tuple().exponent == -6


def test_a_sub_microdollar_cost_rounds_to_zero_and_says_so_by_being_zero() -> None:
    """Recorded honestly as zero-to-six-places, which is what it is."""
    book = PriceBook(tts=Rates(input_usd_per_unit=Decimal("0.0000000001")))
    usage = tts_usage("elevenlabs", 1)

    assert usage is not None
    assert price(usage, book).cost_usd == Decimal("0.000000")


def test_rounding_is_half_up_and_deterministic() -> None:
    book = PriceBook(tts=Rates(input_usd_per_unit=Decimal("0.0000005")))
    usage = tts_usage("elevenlabs", 1)

    assert usage is not None
    assert price(usage, book).cost_usd == Decimal("0.000001")


def test_the_same_input_always_costs_the_same(priced_settings: Settings) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = llm_usage("anthropic", "some-model", 12_345, 678)

    assert usage is not None
    assert price(usage, book).cost_usd == price(usage, book).cost_usd


def test_a_cost_too_large_to_store_is_recorded_unpriced() -> None:
    """Better a visible gap than a silently truncated amount."""
    book = PriceBook(tts=Rates(input_usd_per_unit=Decimal("1")))
    usage = Usage(
        component=CostComponent.TTS,
        provider="elevenlabs",
        input_units=Decimal("100000"),
    )

    costed = price(usage, book)
    assert costed.cost_usd is None
    assert costed.detail["unpriced_reason"] == "cost_out_of_range"


def test_a_cost_at_the_storage_limit_is_kept() -> None:
    book = PriceBook(tts=Rates(input_usd_per_unit=Decimal("1")))
    usage = Usage(
        component=CostComponent.TTS,
        provider="elevenlabs",
        input_units=Decimal("9999"),
    )

    assert price(usage, book).cost_usd == Decimal("9999.000000")


def test_the_storage_limit_matches_the_column(priced_settings: Settings) -> None:
    """`Numeric(10, 6)` holds six figures before the point and six after."""
    assert MAX_COST_USD == Decimal("9999.999999")


def test_zero_units_at_a_real_rate_cost_zero(priced_settings: Settings) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = tts_usage("elevenlabs", 0)

    assert usage is not None
    assert price(usage, book).cost_usd == Decimal("0.000000")


def test_the_line_is_always_unpriced_when_costed(priced_settings: Settings) -> None:
    book = PriceBook.from_settings(priced_settings)
    usage = telephony_usage("twilio", 120_000)

    assert usage is not None
    costed = price(usage, book)
    assert costed.cost_usd is None
    assert costed.detail["unpriced_reason"] == "no_price_configured"


# --- the module's shape ---------------------------------------------------


def test_only_this_module_knows_a_rate() -> None:
    """Nothing else in the application computes money from units."""
    importers = sorted(
        module.relative_to(APP).as_posix()
        for module in APP.rglob("*.py")
        if any(
            isinstance(node, ast.ImportFrom) and node.module == "app.cost.pricing"
            for node in ast.walk(ast.parse(module.read_text()))
        )
    )

    assert importers == ["cost/__init__.py", "cost/recorder.py"]


def test_the_pricing_module_reaches_no_network_and_no_vendor() -> None:
    roots = {
        name.split(".")[0]
        for node in ast.walk(ast.parse(PRICING.read_text()))
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module]
            if isinstance(node, ast.ImportFrom) and node.module
            else []
        )
    }

    assert not roots & {"httpx", "requests", "urllib", "socket", "anthropic"}
