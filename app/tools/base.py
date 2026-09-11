"""Shared pieces of the tool layer.

A tool is a plain function the future dialogue layer will call. It does three
things and no more: normalise the arguments it was handed, delegate to
`CalendarService`, and describe the outcome. It contains no calendar,
business-hours, availability or concurrency logic of its own — all of that
lives in `app/calendar/` and is reached through the service.

Tools **return** failures rather than raising them. A model that asks for a
slot someone else has taken must be told so in a way it can recover from; an
exception would end the turn instead. `tool_calls.success` and
`tool_calls.error` in the data model are shaped for exactly this, and
milestone 4 writes those rows — milestone 3 tools persist nothing themselves,
because `tool_calls.turn_id` is required and turns do not exist yet.
"""

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.calendar import CalendarService
from app.config import Settings, get_settings
from app.models import Service


@dataclass(frozen=True)
class ToolResult:
    """What a tool reports back.

    `data` is plain JSON-safe values so milestone 4 can write it straight into
    `tool_calls.result`; `error` maps to `tool_calls.error`.
    """

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @classmethod
    def ok(cls, **data: Any) -> "ToolResult":
        return cls(success=True, data=data)

    @classmethod
    def failed(cls, error: str, **data: Any) -> "ToolResult":
        return cls(success=False, data=data, error=error)

    def as_dict(self) -> dict[str, Any]:
        return {"success": self.success, "data": self.data, "error": self.error}


@dataclass(frozen=True)
class ToolContext:
    """What every tool needs, and nothing a model gets to choose.

    `call_id` is ambient for the conversation rather than an argument, so a
    model cannot attribute a booking to a call that is not its own.
    """

    session: Session
    settings: Settings | None = None
    call_id: uuid.UUID | None = None

    def calendar(self) -> CalendarService:
        return CalendarService(self.session, self.settings or get_settings())

    def timezone(self) -> ZoneInfo:
        return ZoneInfo((self.settings or get_settings()).business_timezone)


class ToolArgumentError(Exception):
    """An argument could not be understood. Caught and returned, never raised out."""


# --- argument normalisation -----------------------------------------------


def parse_day(value: date | str, field_name: str = "day") -> date:
    """An ISO date, or a date that is already one."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ToolArgumentError(
            f"{field_name} must be an ISO date such as 2026-03-02, not {value!r}."
        ) from exc


def parse_instant(
    value: datetime | str, timezone: ZoneInfo, field_name: str = "starts_at"
) -> datetime:
    """An ISO timestamp, resolved to an instant.

    A value with no offset is read as a wall-clock time in the configured
    business timezone — the mirror of how availability is reported. "Ten
    o'clock" from a caller means ten o'clock where the business is, and a
    timestamp with no offset names no instant on its own.
    """
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip())
        except ValueError as exc:
            raise ToolArgumentError(
                f"{field_name} must be an ISO timestamp such as "
                f"2026-03-02T10:00:00, not {value!r}."
            ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone)
    return parsed


def parse_identifier(value: uuid.UUID | str, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value).strip())
    except ValueError as exc:
        raise ToolArgumentError(
            f"{field_name} must be an identifier, not {value!r}."
        ) from exc


def require_text(value: str, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ToolArgumentError(f"{field_name} is required.")
    return text


# --- service resolution ----------------------------------------------------


def offered_service_names(session: Session) -> list[str]:
    """Every active service, so a failure can say what is actually offered."""
    return list(
        session.execute(
            select(Service.name).where(Service.active.is_(True)).order_by(Service.name)
        )
        .scalars()
        .all()
    )


def resolve_service(session: Session, service_name: str) -> Service:
    """Find the one active service with this name, or fail loudly.

    Trimmed, case-insensitive, exact. No fuzzy matching and no guessing: if
    two active services share a name the caller is told it is ambiguous rather
    than having one picked for them.
    """
    wanted = require_text(service_name, "service_name")

    matches = list(
        session.execute(
            select(Service)
            .where(Service.active.is_(True))
            .where(func.lower(Service.name) == wanted.lower())
            .order_by(Service.name)
        )
        .scalars()
        .all()
    )

    if not matches:
        raise ServiceLookupError(
            f"There is no service called {wanted!r}.",
            offered=offered_service_names(session),
        )
    if len(matches) > 1:
        raise ServiceLookupError(
            f"{wanted!r} matches more than one service, so it is ambiguous.",
            offered=offered_service_names(session),
        )
    return matches[0]


class ServiceLookupError(Exception):
    """The service name did not resolve to exactly one active service."""

    def __init__(self, message: str, offered: list[str]) -> None:
        super().__init__(message)
        self.offered = offered
