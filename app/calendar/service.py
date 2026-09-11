"""The calendar service: the only way anything books, moves or cancels.

    future tool / dialogue layer
              ↓
        CalendarService
              ↓
         PostgreSQL

Two rules hold throughout:

* **Availability is computed here, never guessed.** A later dialogue layer
  must ask this service what is free; it may not invent a time.
* **The database is the authority on conflicts.** `is_available` is advisory
  and can be stale by the time anyone acts on it, so `book` and `reschedule`
  do not check-then-insert. They write and let the exclusion constraint refuse
  them, turning that refusal into `SlotUnavailable`. Two callers racing for
  one slot therefore cannot both win, however the timing falls.
"""

import uuid
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.calendar import availability, hours
from app.calendar.availability import Slot
from app.calendar.errors import (
    AppointmentAlreadyCancelled,
    AppointmentNotFound,
    AppointmentNotReschedulable,
    InvalidInterval,
    InvalidServiceDuration,
    OutsideBusinessHours,
    ServiceNotFound,
    SlotUnavailable,
)
from app.config import Settings, get_settings
from app.models import Appointment, AppointmentStatus, Service


class CalendarService:
    """Calendar operations against one session and one configuration."""

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._timezone = ZoneInfo(self._settings.business_timezone)
        self._granularity = timedelta(
            minutes=self._settings.slot_granularity_minutes
        )

    # --- reading -----------------------------------------------------------

    @property
    def timezone(self) -> ZoneInfo:
        return self._timezone

    def available_slots(self, service_id: uuid.UUID, day: date) -> list[Slot]:
        """Bookable start times for a service on one local day, earliest first."""
        service = self._bookable_service(service_id)
        return availability.free_slots(
            self._session,
            staff_id=service.staff_id,
            day=day,
            duration=self._duration(service),
            granularity=self._granularity,
            timezone=self._timezone,
        )

    def is_available(self, service_id: uuid.UUID, starts_at: datetime) -> bool:
        """Could this service start then?

        Advisory. A `True` here does not reserve anything — only `book` does,
        and only the database can settle a race.
        """
        service = self._bookable_service(service_id)
        ends_at = self._interval(service, starts_at)[1]

        if not hours.fits_in_one_period(
            self._session, starts_at, ends_at, self._timezone
        ):
            return False
        return availability.is_free(
            self._session,
            staff_id=service.staff_id,
            starts_at=starts_at,
            ends_at=ends_at,
        )

    # --- writing -----------------------------------------------------------

    def book(
        self,
        *,
        service_id: uuid.UUID,
        starts_at: datetime,
        customer_name: str,
        phone: str,
        call_id: uuid.UUID | None = None,
    ) -> Appointment:
        """Create an appointment, or say why it could not be created.

        No availability check runs first, deliberately: it would be a lie the
        moment another transaction committed. The write is attempted and the
        constraint decides.
        """
        service = self._bookable_service(service_id)
        starts_at, ends_at = self._interval(service, starts_at)
        self._require_open(starts_at, ends_at)

        appointment = Appointment(
            customer_name=customer_name,
            phone=phone,
            service_id=service.id,
            # Copied from the service so the exclusion constraint can scope
            # overlap to this staff member's diary.
            staff_id=service.staff_id,
            starts_at=starts_at,
            ends_at=ends_at,
            status=AppointmentStatus.BOOKED,
            call_id=call_id,
        )
        self._write(appointment, starts_at, ends_at, service.staff_id)
        return appointment

    def reschedule(
        self, appointment_id: uuid.UUID, new_starts_at: datetime
    ) -> Appointment:
        """Move an existing appointment, keeping it the same appointment.

        The row is updated in place, so its id, creation time and originating
        call survive: a rescheduled booking is the same commitment at a new
        time, not a new one.
        """
        appointment = self._appointment(appointment_id)
        if appointment.status is not AppointmentStatus.BOOKED:
            raise AppointmentNotReschedulable(
                f"Appointment {appointment_id} is {appointment.status} and "
                "cannot be moved."
            )

        service = self._bookable_service(appointment.service_id)
        new_starts_at, new_ends_at = self._interval(service, new_starts_at)
        self._require_open(new_starts_at, new_ends_at)

        appointment.starts_at = new_starts_at
        appointment.ends_at = new_ends_at
        self._write(appointment, new_starts_at, new_ends_at, appointment.staff_id)
        return appointment

    def cancel(self, appointment_id: uuid.UUID) -> Appointment:
        """Cancel an appointment, freeing its slot.

        The row is kept rather than deleted — the call that made the booking
        still points at it — and the exclusion constraint is partial, so a
        cancelled appointment stops holding its interval.
        """
        appointment = self._appointment(appointment_id)
        if appointment.status is AppointmentStatus.CANCELLED:
            raise AppointmentAlreadyCancelled(
                f"Appointment {appointment_id} is already cancelled."
            )

        appointment.status = AppointmentStatus.CANCELLED
        self._session.commit()
        return appointment

    # --- internals ---------------------------------------------------------

    def _bookable_service(self, service_id: uuid.UUID) -> Service:
        service = self._session.get(Service, service_id)
        if service is None:
            raise ServiceNotFound(f"No service with id {service_id}.")
        if not service.active:
            raise ServiceNotFound(f"Service {service.name!r} is not active.")
        return service

    def _appointment(self, appointment_id: uuid.UUID) -> Appointment:
        appointment = self._session.get(Appointment, appointment_id)
        if appointment is None:
            raise AppointmentNotFound(f"No appointment with id {appointment_id}.")
        return appointment

    def _duration(self, service: Service) -> timedelta:
        if service.duration_minutes <= 0:
            raise InvalidServiceDuration(
                f"Service {service.name!r} has a duration of "
                f"{service.duration_minutes} minutes."
            )
        return timedelta(minutes=service.duration_minutes)

    def _interval(
        self, service: Service, starts_at: datetime
    ) -> tuple[datetime, datetime]:
        """The full interval a service occupies from `starts_at`."""
        if starts_at.tzinfo is None or starts_at.utcoffset() is None:
            raise InvalidInterval(
                "starts_at must be timezone-aware; a wall-clock time on its "
                "own does not name an instant."
            )
        return starts_at, starts_at + self._duration(service)

    def _require_open(self, starts_at: datetime, ends_at: datetime) -> None:
        if not hours.fits_in_one_period(
            self._session, starts_at, ends_at, self._timezone
        ):
            raise OutsideBusinessHours(
                f"{starts_at.isoformat()}–{ends_at.isoformat()} does not fit "
                "inside a single opening period."
            )

    def _write(
        self,
        appointment: Appointment,
        starts_at: datetime,
        ends_at: datetime,
        staff_id: str,
    ) -> None:
        """Persist, translating the exclusion constraint into a domain error.

        The flush happens inside a savepoint so a refusal rolls back only this
        write and leaves the session usable.
        """
        try:
            with self._session.begin_nested():
                self._session.add(appointment)
                self._session.flush()
        except IntegrityError as exc:
            if isinstance(exc.orig, psycopg.errors.ExclusionViolation):
                raise SlotUnavailable(
                    f"{staff_id} is already booked between "
                    f"{starts_at.isoformat()} and {ends_at.isoformat()}."
                ) from exc
            raise

        self._session.commit()
