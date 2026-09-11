"""`appointments` — and the constraint that makes double-booking impossible.

The specification is explicit: "Double-booking is the failure that ends a
client relationship. Enforce it with a database constraint, not application
logic alone." So the database, not the booking code, is the authority here.

Two overlapping bookings for the same staff member cannot both exist. That is
a PostgreSQL exclusion constraint over a time range, not something application
code can guarantee — two concurrent callers can both pass an availability
check and both insert. The constraint makes the second insert fail no matter
how the race is timed.

    EXCLUDE USING gist (
        staff_id WITH =,
        tstzrange(starts_at, ends_at, '[)') WITH &&
    ) WHERE (status = 'booked')

Three decisions that constraint forces:

* **`staff_id` is on this table**, duplicated from the service. An exclusion
  constraint can only reference columns of its own table, and the thing that
  must not overlap is a *person's* diary — one stylist offering two services
  must not be booked for both at once. Keeping it in step with
  `services.staff_id` is the booking layer's job in milestone 2.
* **The range is half-open** (`[)`), so an appointment ending at 10:00 and one
  starting at 10:00 do not overlap.
* **The constraint is partial**, applying only to `booked` rows, so a
  cancelled appointment stops holding its slot.

No booking behaviour is implemented here — that is milestone 2. This is the
shape the data has to have for milestone 2 to be safe.
"""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.call import Call
    from app.models.service import Service


class AppointmentStatus(enum.StrEnum):
    """Only what the v1 tools can produce.

    `reschedule` moves an existing booking's times and leaves it `BOOKED`; it
    is not a third state. A `CANCELLED` appointment is kept rather than
    deleted, so the call that cancelled it still points at something.
    """

    BOOKED = "booked"
    CANCELLED = "cancelled"


# Named here so the migration and the model refer to the same constraint.
NO_DOUBLE_BOOKING = "no_double_booking"


class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        CheckConstraint("ends_at > starts_at", name="ends_after_starts"),
        # The exclusion constraint itself is added by the migration:
        # SQLAlchemy's schema layer has no construct for EXCLUDE, and it needs
        # the btree_gist extension the migration installs.
        Index("ix_appointments_staff_id_starts_at", "staff_id", "starts_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # PII, in named columns so redaction is easy to add later.
    customer_name: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(32))

    service_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("services.id", ondelete="RESTRICT"), index=True
    )
    # Denormalised from the service so the exclusion constraint can scope
    # overlap to one person's diary. See the module docstring.
    staff_id: Mapped[str] = mapped_column(String(64))

    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    status: Mapped[AppointmentStatus] = mapped_column(
        Enum(
            AppointmentStatus,
            name="appointment_status",
            values_callable=lambda members: [member.value for member in members],
        ),
        default=AppointmentStatus.BOOKED,
        server_default=AppointmentStatus.BOOKED.value,
    )

    # The call that made the booking. Nullable, and SET NULL on delete: an
    # appointment outlives the conversation that created it.
    call_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("calls.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    service: Mapped["Service"] = relationship(back_populates="appointments")
    call: Mapped["Call | None"] = relationship(back_populates="appointments")

    def __repr__(self) -> str:
        return (
            f"<Appointment {self.customer_name!r} staff={self.staff_id} "
            f"{self.starts_at}–{self.ends_at} {self.status}>"
        )
