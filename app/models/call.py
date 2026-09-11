"""The record of one phone call: `calls`, `turns` and `tool_calls`.

One aggregate. A call has turns in order; a turn may have made tool calls.
The specification requires every transcript to be stored, so these tables are
the durable record of what was said and what the agent did about it.

PII lives in named columns (`from_number`, `to_number`) so that redaction can
be added later without hunting through free text.
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.appointment import Appointment


class CallDirection(enum.StrEnum):
    """Which way the call went.

    v1 only ever writes `INBOUND`: outbound calling and diallers are an
    explicit non-goal. Both values exist because the column is `direction`
    and a direction has two of them.
    """

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallOutcome(enum.StrEnum):
    """How the call ended. Null until it has."""

    BOOKED = "booked"
    RESCHEDULED = "rescheduled"
    CANCELLED = "cancelled"
    MESSAGE_TAKEN = "message_taken"
    TRANSFERRED = "transferred"
    ABANDONED = "abandoned"
    FAILED = "failed"


class TurnRole(enum.StrEnum):
    CALLER = "caller"
    AGENT = "agent"


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    direction: Mapped[CallDirection] = mapped_column(
        Enum(
            CallDirection,
            name="call_direction",
            values_callable=lambda members: [member.value for member in members],
        ),
        default=CallDirection.INBOUND,
        server_default=CallDirection.INBOUND.value,
    )
    # PII. E.164 comfortably fits in 32 characters.
    from_number: Mapped[str] = mapped_column(String(32))
    to_number: Mapped[str] = mapped_column(String(32))

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    # Null while the call is still in progress.
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Null until the call ends; a call in progress has no outcome yet.
    outcome: Mapped[CallOutcome | None] = mapped_column(
        Enum(
            CallOutcome,
            name="call_outcome",
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=True,
    )
    # Computed after the call from speech, model and telephony usage. Null
    # until then, and null forever if any component reported no usage —
    # milestone 8 fills this in.
    total_cost_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 6), nullable=True
    )

    turns: Mapped[list["Turn"]] = relationship(
        back_populates="call",
        cascade="all, delete-orphan",
        order_by="Turn.created_at",
    )
    appointments: Mapped[list["Appointment"]] = relationship(back_populates="call")

    def __repr__(self) -> str:
        return f"<Call {self.id} {self.direction} from={self.from_number!r}>"


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[TurnRole] = mapped_column(
        Enum(
            TurnRole,
            name="turn_role",
            values_callable=lambda members: [member.value for member in members],
        )
    )
    # What was said. Empty rather than null when nothing was recognised: a
    # turn row always represents something that happened.
    text: Mapped[str] = mapped_column(Text)

    audio_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Each latency is null on the turns it cannot apply to: a caller turn has
    # no LLM or TTS latency, an agent turn has no STT latency.
    stt_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tts_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    call: Mapped["Call"] = relationship(back_populates="turns")
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="turn", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Turn {self.role} {self.text[:40]!r}>"


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    turn_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("turns.id", ondelete="CASCADE"), index=True
    )
    tool_name: Mapped[str] = mapped_column(String(64))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Null when the call failed: there is no result to record, only an error.
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    turn: Mapped["Turn"] = relationship(back_populates="tool_calls")

    def __repr__(self) -> str:
        return f"<ToolCall {self.tool_name} success={self.success}>"
