"""The cost layer: what a call consumed, and what that cost if anyone knows.

    usage.py     what was consumed, in units, with no opinion about money
    pricing.py   the only module that knows a price — and it ships none
    recorder.py  the only thing that writes a cost row or totals a call

VoiceDesk ships no vendor prices. Usage is measured and always recorded;
`cost_usd` is null unless an operator configured a price for that component.
Unknown is never rounded down to free, and a call whose components are not all
priced has no total at all.

This is accounting, not billing. There are no invoices, no quotas, no
customers and no payment processing here, and none are coming.
"""

from app.cost.pricing import PriceBook, Priced, PricingError, Rates, price
from app.cost.recorder import record_call_cost, record_turn_cost, recompute_total
from app.cost.usage import (
    Usage,
    UsageError,
    llm_usage,
    stt_usage,
    telephony_usage,
    tts_usage,
)

__all__ = [
    "PriceBook",
    "Priced",
    "PricingError",
    "Rates",
    "Usage",
    "UsageError",
    "llm_usage",
    "price",
    "recompute_total",
    "record_call_cost",
    "record_turn_cost",
    "stt_usage",
    "telephony_usage",
    "tts_usage",
]
