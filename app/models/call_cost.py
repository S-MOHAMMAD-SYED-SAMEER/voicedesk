"""What one call consumed, and what that cost — when the price is known.

    calls ──┬── call_costs (component = llm)     ── one per agent turn
            ├── call_costs (component = stt)     ── one per agent turn
            ├── call_costs (component = tts)     ── one per agent turn
            └── call_costs (component = telephony) ── one per call

Two columns of this table are deliberately different from the rest.

* **`input_units` and `output_units` are always written.** What was consumed
  is something this system measured; it is a fact whether or not anyone can
  price it.
* **`cost_usd` is null unless a price was configured.** VoiceDesk ships no
  vendor prices. An unpriced component is recorded as unpriced, never as
  free — a zero here would be a claim about money that nobody made.

`unit_price_usd` is the one rate that produced `cost_usd`. A component billed
at two rates — a model's input and output tokens — leaves it null and records
both rates in `metadata`, because one column cannot honestly hold two prices.
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Numeric,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CostComponent(enum.StrEnum):
    """Which part of a call a row is about.

    These are the four things a phone call spends: a model, recognition,
    synthesis, and the line itself.
    """

    LLM = "llm"
    STT = "stt"
    TTS = "tts"
    TELEPHONY = "telephony"


# Enforced by a partial unique index, because PostgreSQL treats nulls in a
# unique constraint as distinct: `(turn_id, component)` alone would happily
# accept a second telephony row for the same call.
TURN_COMPONENT_UNIQUE = "uq_call_costs_turn_component"
CALL_COMPONENT_UNIQUE = "uq_call_costs_call_component_without_turn"


class CallCost(Base):
    __tablename__ = "call_costs"

    __table_args__ = (
        Index(TURN_COMPONENT_UNIQUE, "turn_id", "component", unique=True),
        # One row per call for a component that has no turn. Partial, so it
        # constrains exactly the rows the constraint above cannot reach.
        Index(
            CALL_COMPONENT_UNIQUE,
            "call_id",
            "component",
            unique=True,
            postgresql_where=text("turn_id IS NULL"),
        ),
        Index("ix_call_costs_call_id_component", "call_id", "component"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    # Null for a component that belongs to the whole call. `SET NULL` rather
    # than `CASCADE`: if a turn is ever deleted, what it spent still happened
    # and the call still owns it.
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("turns.id", ondelete="SET NULL"), nullable=True
    )
    component: Mapped[CostComponent] = mapped_column(
        Enum(
            CostComponent,
            name="cost_component",
            values_callable=lambda members: [member.value for member in members],
        )
    )
    # Who did the work, and — for a model — which one. Recorded because a
    # price only means anything alongside what it was a price for.
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Measured, never inferred. `unit_type` says what they count: "tokens",
    # "audio_ms", "characters" or "duration_ms".
    input_units: Mapped[Decimal] = mapped_column(Numeric(18, 3))
    output_units: Mapped[Decimal] = mapped_column(
        Numeric(18, 3), default=Decimal("0"), server_default=text("0")
    )
    unit_type: Mapped[str] = mapped_column(String(32))

    # Per single unit, not per million of them: the arithmetic that produced
    # `cost_usd` should be checkable by multiplying two columns together.
    unit_price_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 12), nullable=True
    )
    # Null means "not priced", which is not the same as "free".
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)

    # `metadata` is reserved on a declarative class, so the attribute is
    # `details` and the column keeps the name the schema wants.
    details: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<CallCost {self.component} {self.provider} "
            f"cost_usd={self.cost_usd}>"
        )
