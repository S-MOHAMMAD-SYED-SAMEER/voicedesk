"""`services` — what the business offers, and who performs it."""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, Integer, String, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.appointment import Appointment


class Service(Base):
    __tablename__ = "services"
    __table_args__ = (
        CheckConstraint("duration_minutes > 0", name="duration_minutes_positive"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(128))
    duration_minutes: Mapped[int] = mapped_column(Integer)
    # Who performs this service. An opaque identifier: the v1 specification
    # has no staff table, so this is not a foreign key. It is what scopes the
    # double-booking constraint on `appointments`.
    staff_id: Mapped[str] = mapped_column(String(64), index=True)
    active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=text("true")
    )

    appointments: Mapped[list["Appointment"]] = relationship(back_populates="service")

    def __repr__(self) -> str:
        return f"<Service {self.name!r} {self.duration_minutes}min staff={self.staff_id}>"
