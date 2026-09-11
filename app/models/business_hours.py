"""`business_hours` — when the business is open.

One row per opening period. A day with a lunch break is two rows for that
weekday, so no uniqueness is imposed on `weekday`.
"""

import uuid
from datetime import time

from sqlalchemy import CheckConstraint, SmallInteger, Time, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# ISO 8601 weekday numbering, matching `datetime.date.weekday()`.
MONDAY = 0
SUNDAY = 6


class BusinessHours(Base):
    __tablename__ = "business_hours"
    __table_args__ = (
        CheckConstraint(
            f"weekday BETWEEN {MONDAY} AND {SUNDAY}", name="weekday_in_range"
        ),
        CheckConstraint("closes_at > opens_at", name="closes_after_opens"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    # 0 = Monday … 6 = Sunday, the same numbering as `date.weekday()`.
    weekday: Mapped[int] = mapped_column(SmallInteger, index=True)
    # Wall-clock times, not instants: "we open at nine" is not a moment.
    opens_at: Mapped[time] = mapped_column(Time)
    closes_at: Mapped[time] = mapped_column(Time)

    def __repr__(self) -> str:
        return f"<BusinessHours weekday={self.weekday} {self.opens_at}-{self.closes_at}>"
