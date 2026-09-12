"""The only thing that writes a cost row, and the only thing that totals one.

    turn finishes ─► record_turn_cost ─► call_costs (llm, stt, tts)
    call ends     ─► record_call_cost ─► call_costs (telephony)
                                              │
                                              └─► calls.total_cost_usd

Three rules hold here, and each exists because the alternative would produce
a number somebody might believe.

* **Usage is written whether or not it can be priced.** What was consumed is
  measured; what it costs may be unknown. An unpriced row carries its units
  and a null `cost_usd`.
* **One unpriced component makes the whole call unpriced.** `total_cost_usd`
  is null unless every row of that call has a cost. A partial total looks
  exactly like a complete one, and would be read as a complete one.
* **Recording twice records once.** Rows are keyed by turn and component (and
  by call and component for the telephony row), so a retried write updates
  the row it already made instead of doubling the call's spend.

The total is recomputed with a single aggregate over `call_costs`, never by
adding to whatever the column happened to hold. Read-modify-write on a column
two turns can finish at the same time is how totals drift.

Best effort throughout, like the latency writer next door: a call is not
worth failing over an accounting row.
"""

import logging
import uuid
from decimal import Decimal, ROUND_HALF_UP
from typing import Protocol

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.cost.pricing import CENTS, MAX_COST_USD, PriceBook, price
from app.cost.usage import (
    Usage,
    llm_usage,
    stt_usage,
    telephony_usage,
    tts_usage,
)
from app.models import Call, CallCost, CostComponent, Turn

logger = logging.getLogger(__name__)


class HasDialogueUsage(Protocol):
    """The part of a dialogue result this module needs."""

    caller_turn_id: object
    agent_turn_id: object
    model_name: str
    input_tokens: int | None
    output_tokens: int | None


class HasSpeechUsage(Protocol):
    """The part of a turn's speech this module needs."""

    stt_provider: str
    stt_audio_ms: int | None
    tts_provider: str
    tts_characters: int | None


def record_turn_cost(
    session: Session,
    result: HasDialogueUsage,
    speech: HasSpeechUsage,
    settings: Settings | None = None,
    *,
    llm_provider: str = "",
) -> None:
    """Write what one turn consumed, and re-total the call.

    `llm_provider` is passed in rather than worked out here: which vendor
    answered is the transport's to know, through the factory that chose it.
    Nothing in this package names a vendor.
    """
    resolved = settings or get_settings()
    if not resolved.cost_tracking_enabled:
        return

    try:
        _write_turn(session, result, speech, resolved, llm_provider)
    except Exception:  # noqa: BLE001 - accounting must not break a call
        logger.exception("Could not record the cost of a turn.")
        session.rollback()


def record_call_cost(
    session: Session,
    call: Call,
    provider: str,
    settings: Settings | None = None,
) -> None:
    """Write the line's own usage for a finished call, and re-total it.

    Recorded unpriced, always. See `app/cost/pricing.py` for why measured
    wall-clock time is not the thing a carrier bills for.
    """
    resolved = settings or get_settings()
    if not resolved.cost_tracking_enabled:
        return

    try:
        _write_call(session, call, provider, resolved)
    except Exception:  # noqa: BLE001 - accounting must not break a call
        logger.exception("Could not record the cost of call %s.", call.id)
        session.rollback()


def recompute_total(session: Session, call_id: uuid.UUID) -> Decimal | None:
    """Re-derive `calls.total_cost_usd` from the rows, and store it.

    Returns what was stored, which is `None` when the call has no cost rows
    at all or when any one of them could not be priced.
    """
    total, unpriced, rows = session.execute(
        select(
            func.sum(CallCost.cost_usd),
            func.count().filter(CallCost.cost_usd.is_(None)),
            func.count(),
        ).where(CallCost.call_id == call_id)
    ).one()

    resolved: Decimal | None
    if rows == 0 or unpriced or total is None:
        resolved = None
    else:
        resolved = Decimal(total).quantize(CENTS, rounding=ROUND_HALF_UP)
        if resolved > MAX_COST_USD:
            logger.error(
                "Call %s totals %s, which cannot be stored; leaving it null.",
                call_id,
                resolved,
            )
            resolved = None

    session.execute(
        update(Call).where(Call.id == call_id).values(total_cost_usd=resolved)
    )
    session.commit()
    return resolved


# --- internals -------------------------------------------------------------


def _write_turn(
    session: Session,
    result: HasDialogueUsage,
    speech: HasSpeechUsage,
    settings: Settings,
    llm_provider: str,
) -> None:
    agent_turn_id = result.agent_turn_id
    caller_turn_id = result.caller_turn_id
    if agent_turn_id is None:
        return

    call_id = _call_of_turn(session, agent_turn_id)
    if call_id is None:
        logger.warning("No call owns turn %s; recording no cost.", agent_turn_id)
        return

    book = PriceBook.from_settings(settings)

    # The model and the synthesis belong to the reply; the audio that was
    # recognised belongs to the caller's own turn, which is where `audio_ms`
    # already lives.
    pairs: list[tuple[object | None, Usage | None]] = [
        (
            agent_turn_id,
            llm_usage(
                provider=llm_provider,
                model=result.model_name or None,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            if llm_provider
            else None,
        ),
        (
            caller_turn_id or agent_turn_id,
            stt_usage(speech.stt_provider, speech.stt_audio_ms)
            if speech.stt_provider
            else None,
        ),
        (
            agent_turn_id,
            tts_usage(speech.tts_provider, speech.tts_characters)
            if speech.tts_provider
            else None,
        ),
    ]

    wrote = False
    for turn_id, usage in pairs:
        if usage is None:
            continue
        _upsert(session, call_id, turn_id, usage, book)
        wrote = True

    if not wrote:
        return
    session.commit()
    recompute_total(session, call_id)


def _write_call(
    session: Session, call: Call, provider: str, settings: Settings
) -> None:
    if not provider:
        return
    if call.ended_at is None or call.started_at is None:
        # A call still in progress has no duration to record.
        return

    elapsed = call.ended_at - call.started_at
    duration_ms = int(elapsed.total_seconds() * 1000)
    if duration_ms < 0:
        logger.warning(
            "Call %s ended before it started; recording no line usage.",
            call.id,
        )
        return

    usage = telephony_usage(provider, duration_ms)
    if usage is None:
        return

    _upsert(session, call.id, None, usage, PriceBook.from_settings(settings))
    session.commit()
    recompute_total(session, call.id)


def _upsert(
    session: Session,
    call_id: uuid.UUID,
    turn_id: object | None,
    usage: Usage,
    book: PriceBook,
) -> CallCost:
    """One row per (turn, component), or per (call, component) without a turn."""
    costed = price(usage, book)
    row = _existing(session, call_id, turn_id, usage.component)

    if row is None:
        row = CallCost(call_id=call_id, turn_id=turn_id, component=usage.component)
        session.add(row)

    row.provider = usage.provider
    row.model = usage.model
    row.input_units = usage.input_units
    row.output_units = usage.output_units
    row.unit_type = usage.unit_type
    row.unit_price_usd = costed.unit_price_usd
    row.cost_usd = costed.cost_usd
    row.details = {**usage.metadata, **costed.detail}
    return row


def _existing(
    session: Session,
    call_id: uuid.UUID,
    turn_id: object | None,
    component: CostComponent,
) -> CallCost | None:
    query = select(CallCost).where(CallCost.component == component)
    if turn_id is None:
        # `turn_id IS NULL` rows are unique per call, not globally: a unique
        # index cannot see two nulls as equal, so the partial index does it.
        query = query.where(
            CallCost.call_id == call_id, CallCost.turn_id.is_(None)
        )
    else:
        query = query.where(CallCost.turn_id == turn_id)
    return session.execute(query).scalar_one_or_none()


def _call_of_turn(session: Session, turn_id: object) -> uuid.UUID | None:
    return session.execute(
        select(Turn.call_id).where(Turn.id == turn_id)
    ).scalar_one_or_none()


__all__ = [
    "HasDialogueUsage",
    "HasSpeechUsage",
    "recompute_total",
    "record_call_cost",
    "record_turn_cost",
]
