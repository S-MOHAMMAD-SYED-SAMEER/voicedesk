"""Finding a caller's existing bookings.

Matching is by **name and telephone number together**: a booking is returned only when both agree. A number is
compared by its last nine digits, so "07700 900123" and "+44 7700 900123" are the same number; a name is compared
case-insensitively and exactly. Nothing else is fuzzy.

This is the identity check behind rescheduling and cancelling — the dialogue layer lets a model change only an
appointment it has been handed by this lookup or by a booking made in the same call. It is not strong authentication:
anyone who knows a customer's name and number can find that customer's bookings. It does stop a caller reaching a
booking by guessing.
"""

import re
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Appointment, AppointmentStatus

_NON_DIGITS = re.compile(r"\D")
# Enough digits to tell two numbers apart, few enough that a country code or a leading zero does not matter.
SIGNIFICANT_DIGITS = 9
MINIMUM_DIGITS = 6
DEFAULT_LIMIT = 10


def phone_key(number: str) -> str:
    """The part of a telephone number that identifies it: its last nine digits."""
    return _NON_DIGITS.sub("", number)[-SIGNIFICANT_DIGITS:]


def upcoming_for(
    session: Session,
    *,
    customer_name: str,
    phone: str,
    after: datetime,
    limit: int = DEFAULT_LIMIT,
) -> list[Appointment]:
    """Booked appointments starting after `after` whose name and number both match, earliest first."""
    digits = func.regexp_replace(Appointment.phone, r"\D", "", "g")
    return list(
        session.execute(
            select(Appointment)
            .where(Appointment.status == AppointmentStatus.BOOKED)
            .where(Appointment.starts_at > after)
            .where(func.lower(Appointment.customer_name) == customer_name.strip().lower())
            .where(func.right(digits, SIGNIFICANT_DIGITS) == phone_key(phone))
            .order_by(Appointment.starts_at)
            .limit(limit)
        )
        .scalars()
        .all()
    )
