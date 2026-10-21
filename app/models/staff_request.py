"""Something a person on the staff has to deal with.

A receptionist that cannot connect a caller to anyone must still not lose what the caller wanted. Three things end up
here, each one row:

* `message`   — the caller left a message for the team (`take_message`).
* `transfer`  — the caller asked to speak to a person, or the conversation reached something a person should handle
                (`transfer_to_human`). **No live transfer exists**: this row is the request that somebody call back.
* `failure`   — the system could not answer (the model was unavailable, or the tool loop ran out), and the caller was
                told the team would call back. Recorded so that promise has something behind it.

The rows keep the call they came from, and the caller's number as the caller gave it or, failing that, as the carrier
reported it. `handled_at` is for a person to fill in; nothing in the application writes it.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, Uuid, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class StaffRequestKind(enum.StrEnum):
    MESSAGE = "message"
    TRANSFER = "transfer"
    FAILURE = "failure"


class StaffRequest(Base):
    __tablename__ = "staff_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # SET NULL: a request outlives the retention of the call it came from.
    call_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("calls.id", ondelete="SET NULL"), nullable=True, index=True
    )
    kind: Mapped[StaffRequestKind] = mapped_column(
        Enum(
            StaffRequestKind,
            name="staff_request_kind",
            values_callable=lambda members: [member.value for member in members],
        )
    )
    # PII, in named columns so redaction is easy to add later.
    caller_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # The message, or the reason a person was asked for.
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    handled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<StaffRequest {self.kind} call={self.call_id}>"
