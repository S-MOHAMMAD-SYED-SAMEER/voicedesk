"""What a call consumed, in units, with no opinion about money.

This module is deliberately price-free. It names the four things a call
spends and the units each is counted in; `app/cost/pricing.py` is the only
place that knows what any of it costs, and it may well know nothing.

    tokens        a model's input and output
    audio_ms      how long the recogniser was given
    characters    how much text the synthesiser was given
    duration_ms   how long the line was open

Every builder here returns `None` when the quantity is unknown. That is the
whole discipline of this milestone: a provider that reported nothing produces
no row, rather than a row claiming it consumed nothing.

**Characters are never summed across chunks.** A streaming synthesiser
reports `characters` on each chunk, and every implementation in this
repository reports the *whole* text on every one of them — so adding them up
multiplies the true figure by the number of chunks. `tts_usage` therefore
takes one number, counted once per stream by whoever asked for the speech.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.models import CostComponent

# What `Usage.unit_type` may say, and what each means.
TOKENS = "tokens"
AUDIO_MS = "audio_ms"
CHARACTERS = "characters"
DURATION_MS = "duration_ms"

UNIT_TYPES = (TOKENS, AUDIO_MS, CHARACTERS, DURATION_MS)

# The unit each component is measured in. One component, one unit: a row
# whose units cannot be compared with another row of the same component
# would make the totals meaningless.
UNIT_FOR_COMPONENT = {
    CostComponent.LLM: TOKENS,
    CostComponent.STT: AUDIO_MS,
    CostComponent.TTS: CHARACTERS,
    CostComponent.TELEPHONY: DURATION_MS,
}


class UsageError(ValueError):
    """A quantity that cannot be true: a negative one, most often."""


@dataclass(frozen=True)
class Usage:
    """One component's consumption on one turn, or on one whole call.

    Vendor-neutral by construction: `provider` and `model` are names carried
    through for the record, not switches anything branches on.
    """

    component: CostComponent
    provider: str
    input_units: Decimal
    output_units: Decimal = Decimal("0")
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("input_units", "output_units"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise UsageError(
                    f"{name} must be a Decimal, not {type(value).__name__}; "
                    "floats do not add up to money."
                )
            if value < 0:
                raise UsageError(f"{name} cannot be negative: {value}")
        if not self.provider:
            raise UsageError("A usage record must name who did the work.")

    @property
    def unit_type(self) -> str:
        return UNIT_FOR_COMPONENT[self.component]

    @property
    def total_units(self) -> Decimal:
        return self.input_units + self.output_units


def llm_usage(
    provider: str,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> Usage | None:
    """A model turn, if the provider said what it used.

    Both counts are required. A response that reported only one of them is
    half-measured, and half a measurement priced as a whole one would be
    wrong in a direction nobody could see.
    """
    if input_tokens is None or output_tokens is None:
        return None
    return Usage(
        component=CostComponent.LLM,
        provider=provider,
        model=model,
        input_units=Decimal(int(input_tokens)),
        output_units=Decimal(int(output_tokens)),
        # Cache reads and writes are not measured: the response object this
        # repository reads does not expose them. A cached prompt is therefore
        # priced as if it were not cached, which over-states it.
        metadata={"cache_tokens_measured": False},
    )


def stt_usage(provider: str, audio_ms: int | None) -> Usage | None:
    """Recognition, measured as the audio handed to it.

    Not the same thing a vendor bills. A streaming recogniser is usually
    billed for how long the connection was open, and this counts only the
    audio inside the results it returned — which is less. See the README.
    """
    if audio_ms is None:
        return None
    return Usage(
        component=CostComponent.STT,
        provider=provider,
        input_units=Decimal(int(audio_ms)),
        metadata={"measured": "audio_in_results"},
    )


def tts_usage(provider: str, characters: int | None) -> Usage | None:
    """Synthesis, measured as the text handed to it — counted once.

    `characters` is the length of the text asked for, not a sum over the
    chunks that came back. Every streaming implementation here repeats the
    full count on every chunk, so summing would multiply it.
    """
    if characters is None:
        return None
    return Usage(
        component=CostComponent.TTS,
        provider=provider,
        input_units=Decimal(int(characters)),
        metadata={"counted": "once_per_stream"},
    )


def telephony_usage(provider: str, duration_ms: int | None) -> Usage | None:
    """The line, measured as the call's own start and end.

    This is not billable time. A carrier bills from its own record of the
    call, rounded up to whole units, and this process never sees that record.
    The duration is worth keeping; pricing it would produce a number that
    looked like a bill and was not one.
    """
    if duration_ms is None:
        return None
    return Usage(
        component=CostComponent.TELEPHONY,
        provider=provider,
        input_units=Decimal(int(duration_ms)),
        metadata={"measured": "wall_clock_between_started_at_and_ended_at"},
    )


__all__ = [
    "AUDIO_MS",
    "CHARACTERS",
    "DURATION_MS",
    "TOKENS",
    "UNIT_FOR_COMPONENT",
    "UNIT_TYPES",
    "Usage",
    "UsageError",
    "llm_usage",
    "stt_usage",
    "telephony_usage",
    "tts_usage",
]
