"""The only module in VoiceDesk that knows what anything costs.

**VoiceDesk ships no vendor prices.** Not Anthropic's, not Deepgram's, not
ElevenLabs', not Twilio's. Prices change, this repository cannot check them,
and a stale number baked into source is worse than no number at all: it would
be quoted as a fact for as long as it survived. So the shipped table is empty
of them, and a component is priced only when an operator has put a price in
their own configuration.

The one exception is not an exception to that rule. The `offline` providers
are this repository's own code — a fixed transcript and a tone. They cost
nothing because nothing is bought, which is a fact about this codebase rather
than a claim about anybody's price list, so they are priced at exactly zero
and configuration cannot override them.

Everything else resolves to `None`, and `None` travels all the way to a null
`cost_usd`. Unknown is never rounded down to free.

    rate = what the operator configured, converted to a per-unit price
    cost = rate × units, both quantised, so the arithmetic checks out in SQL

Telephony is never priced here at all, whatever anyone configures. What this
system can measure is the wall-clock window in which a media stream was open;
what a carrier bills is its own record of the call, rounded up to whole units,
which this process never sees. Pricing the first as if it were the second
would produce a number that looked like a bill and was not one.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from app.config import Settings, get_settings
from app.cost.usage import Usage
from app.models import CostComponent

logger = logging.getLogger(__name__)

# The providers implemented in this repository, which buy nothing.
OFFLINE_PROVIDER = "offline"

# Money is stored to the microdollar and rates to the picodollar. Rates are
# quantised before they are multiplied, so `cost_usd = unit_price_usd × units`
# is true of the stored columns and not merely of the intermediate values.
CENTS = Decimal("0.000001")
RATE = Decimal("0.000000000001")
ZERO = Decimal("0")

# What `calls.total_cost_usd` and `call_costs.cost_usd` can hold.
MAX_COST_USD = Decimal("9999.999999")

TOKENS_PER_MTOK = Decimal(1_000_000)
CHARACTERS_PER_MCHAR = Decimal(1_000_000)
MS_PER_MINUTE = Decimal(60_000)

# Absurdity guards, not knowledge of anybody's price list. They exist so that
# a misplaced decimal point or a pasted currency symbol fails at startup
# rather than quietly producing a five-figure call.
MAX_USD_PER_MTOK = Decimal("10000")
MAX_USD_PER_MINUTE = Decimal("100")
MAX_USD_PER_MCHAR = Decimal("10000")


class PricingError(ValueError):
    """A configured price that cannot be used."""


@dataclass(frozen=True)
class Rates:
    """What one unit costs, and — for a model — what an output unit costs."""

    input_usd_per_unit: Decimal
    output_usd_per_unit: Decimal = ZERO

    @property
    def single(self) -> bool:
        """True when one column can honestly hold the whole rate."""
        return self.output_usd_per_unit == ZERO


@dataclass(frozen=True)
class Priced:
    """One usage record, costed — or explicitly not."""

    cost_usd: Decimal | None
    unit_price_usd: Decimal | None
    detail: dict[str, Any]

    @property
    def priced(self) -> bool:
        return self.cost_usd is not None


# The complete pricing table VoiceDesk ships. Every key names a provider whose
# implementation lives in this repository; no vendor appears here, and none
# ever should.
BUILT_IN_RATES: dict[tuple[CostComponent, str], Rates] = {
    (CostComponent.STT, OFFLINE_PROVIDER): Rates(ZERO),
    (CostComponent.TTS, OFFLINE_PROVIDER): Rates(ZERO),
}

# Components an operator may put a price on. Telephony is absent on purpose.
CONFIGURABLE = (CostComponent.LLM, CostComponent.STT, CostComponent.TTS)


def parse_price(
    value: str, *, setting: str, maximum: Decimal
) -> Decimal | None:
    """One configured price, or `None` when the operator supplied none.

    Blank means unset, which means unpriced. It does not mean zero: an
    operator who has not told us a price has not told us the price is nothing.
    """
    text = value.strip()
    if not text:
        return None
    try:
        price = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise PricingError(
            f"{setting} is {value!r}, which is not a number. Give a plain "
            "decimal amount in US dollars, or leave it empty for unpriced."
        ) from exc
    if not price.is_finite():
        raise PricingError(f"{setting} must be a finite amount, not {value!r}.")
    if price < 0:
        raise PricingError(f"{setting} cannot be negative: {value!r}.")
    if price > maximum:
        raise PricingError(
            f"{setting} is {value!r}, which is above the sanity limit of "
            f"{maximum}. Check the decimal point and the units."
        )
    return price


@dataclass(frozen=True)
class PriceBook:
    """The operator's prices, converted to per-unit amounts.

    `None` on any field means that component is unpriced, and stays unpriced:
    nothing here falls back to a guess.
    """

    llm: Rates | None = None
    stt: Rates | None = None
    tts: Rates | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "PriceBook":
        """Read the configured prices, failing loudly on a malformed one."""
        resolved = settings or get_settings()

        llm_input = parse_price(
            resolved.llm_input_usd_per_mtok,
            setting="VOICEDESK_LLM_INPUT_USD_PER_MTOK",
            maximum=MAX_USD_PER_MTOK,
        )
        llm_output = parse_price(
            resolved.llm_output_usd_per_mtok,
            setting="VOICEDESK_LLM_OUTPUT_USD_PER_MTOK",
            maximum=MAX_USD_PER_MTOK,
        )
        if (llm_input is None) != (llm_output is None):
            # Half a model's price is not a price. Charging only for input
            # would under-state every call, invisibly.
            raise PricingError(
                "Configure both VOICEDESK_LLM_INPUT_USD_PER_MTOK and "
                "VOICEDESK_LLM_OUTPUT_USD_PER_MTOK, or neither."
            )

        stt = parse_price(
            resolved.stt_usd_per_minute,
            setting="VOICEDESK_STT_USD_PER_MINUTE",
            maximum=MAX_USD_PER_MINUTE,
        )
        tts = parse_price(
            resolved.tts_usd_per_mchar,
            setting="VOICEDESK_TTS_USD_PER_MCHAR",
            maximum=MAX_USD_PER_MCHAR,
        )

        return cls(
            llm=(
                None
                if llm_input is None or llm_output is None
                else Rates(
                    input_usd_per_unit=_quantise_rate(llm_input / TOKENS_PER_MTOK),
                    output_usd_per_unit=_quantise_rate(llm_output / TOKENS_PER_MTOK),
                )
            ),
            stt=(
                None
                if stt is None
                else Rates(input_usd_per_unit=_quantise_rate(stt / MS_PER_MINUTE))
            ),
            tts=(
                None
                if tts is None
                else Rates(
                    input_usd_per_unit=_quantise_rate(tts / CHARACTERS_PER_MCHAR)
                )
            ),
        )

    def rates_for(self, usage: Usage) -> Rates | None:
        """What this usage is charged at, or `None` if nobody has said.

        The shipped table is consulted first. Its only entries are this
        repository's own free providers, so configuration cannot accidentally
        put a vendor's price on a tone generator.
        """
        built_in = BUILT_IN_RATES.get((usage.component, usage.provider))
        if built_in is not None:
            return built_in
        if usage.component is CostComponent.TELEPHONY:
            # Never, whatever is configured. See the module docstring.
            return None
        return {
            CostComponent.LLM: self.llm,
            CostComponent.STT: self.stt,
            CostComponent.TTS: self.tts,
        }.get(usage.component)


def price(usage: Usage, book: PriceBook) -> Priced:
    """Cost one usage record, or say plainly that it cannot be costed."""
    rates = book.rates_for(usage)
    if rates is None:
        return Priced(
            cost_usd=None,
            unit_price_usd=None,
            detail={"unpriced_reason": "no_price_configured"},
        )

    cost = (
        usage.input_units * rates.input_usd_per_unit
        + usage.output_units * rates.output_usd_per_unit
    )
    cost = cost.quantize(CENTS, rounding=ROUND_HALF_UP)

    if cost > MAX_COST_USD:
        # Refusing to store it is better than an arithmetic error at the
        # database, and far better than a silently truncated amount.
        logger.error(
            "A %s cost of %s exceeds what can be stored; recording it unpriced.",
            usage.component,
            cost,
        )
        return Priced(
            cost_usd=None,
            unit_price_usd=None,
            detail={"unpriced_reason": "cost_out_of_range"},
        )

    if rates.single:
        return Priced(
            cost_usd=cost, unit_price_usd=rates.input_usd_per_unit, detail={}
        )

    # Two rates, one column. The column stays null and both rates are written
    # down beside the figure they produced.
    return Priced(
        cost_usd=cost,
        unit_price_usd=None,
        detail={
            "input_usd_per_unit": str(rates.input_usd_per_unit),
            "output_usd_per_unit": str(rates.output_usd_per_unit),
        },
    )


def _quantise_rate(value: Decimal) -> Decimal:
    return value.quantize(RATE, rounding=ROUND_HALF_UP)


__all__ = [
    "BUILT_IN_RATES",
    "CENTS",
    "CONFIGURABLE",
    "MAX_COST_USD",
    "MAX_USD_PER_MCHAR",
    "MAX_USD_PER_MINUTE",
    "MAX_USD_PER_MTOK",
    "OFFLINE_PROVIDER",
    "PriceBook",
    "Priced",
    "PricingError",
    "Rates",
    "parse_price",
    "price",
]
