"""Argument normalisation and service resolution, on their own."""

import uuid
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.orm import Session

from app.models import Service
from app.tools.base import (
    ServiceLookupError,
    ToolArgumentError,
    ToolContext,
    offered_service_names,
    parse_day,
    parse_identifier,
    parse_instant,
    require_text,
    resolve_service,
)

LONDON = ZoneInfo("Europe/London")


# --- days ------------------------------------------------------------------


def test_an_iso_string_becomes_a_date() -> None:
    assert parse_day("2026-03-02") == date(2026, 3, 2)


def test_surrounding_whitespace_is_tolerated() -> None:
    assert parse_day("  2026-03-02 ") == date(2026, 3, 2)


def test_a_date_passes_through() -> None:
    assert parse_day(date(2026, 3, 2)) == date(2026, 3, 2)


def test_a_datetime_is_reduced_to_its_day() -> None:
    assert parse_day(datetime(2026, 3, 2, 14, tzinfo=UTC)) == date(2026, 3, 2)


def test_prose_is_refused_with_an_example() -> None:
    with pytest.raises(ToolArgumentError) as caught:
        parse_day("next Tuesday")
    assert "2026-03-02" in str(caught.value)


# --- instants --------------------------------------------------------------


def test_an_offset_in_the_input_is_respected() -> None:
    """A caller who names an instant is not second-guessed."""
    parsed = parse_instant("2026-03-02T10:00:00+05:00", LONDON)

    assert parsed.utcoffset() == timedelta(hours=5)
    assert parsed.astimezone(UTC).hour == 5


def test_a_naive_time_is_read_as_local_to_the_business() -> None:
    """"Ten o'clock" names no instant until you say where."""
    parsed = parse_instant("2026-07-02T10:00:00", LONDON)

    assert parsed.tzinfo is LONDON
    # July, so London is on BST and ten local is nine UTC.
    assert parsed.astimezone(UTC).hour == 9


def test_an_aware_datetime_passes_through() -> None:
    given = datetime(2026, 3, 2, 10, tzinfo=UTC)
    assert parse_instant(given, LONDON) is given


def test_prose_is_refused_for_instants_too() -> None:
    with pytest.raises(ToolArgumentError) as caught:
        parse_instant("half past ten", LONDON, "new_starts_at")
    assert "new_starts_at" in str(caught.value)


# --- identifiers and text --------------------------------------------------


def test_a_uuid_string_becomes_a_uuid() -> None:
    identifier = uuid.uuid4()
    assert parse_identifier(str(identifier), "appointment_id") == identifier


def test_a_uuid_passes_through() -> None:
    identifier = uuid.uuid4()
    assert parse_identifier(identifier, "appointment_id") is identifier


def test_something_that_is_not_an_identifier_names_its_field() -> None:
    with pytest.raises(ToolArgumentError) as caught:
        parse_identifier("the ten o'clock one", "appointment_id")
    assert "appointment_id" in str(caught.value)


def test_required_text_is_trimmed() -> None:
    assert require_text("  Ada  ", "customer_name") == "Ada"


@pytest.mark.parametrize("value", ["", "   ", None])
def test_absent_text_is_refused(value) -> None:
    with pytest.raises(ToolArgumentError):
        require_text(value, "customer_name")


# --- service resolution ----------------------------------------------------


def test_an_exact_name_resolves(session: Session, haircut: Service) -> None:
    assert resolve_service(session, "Haircut").id == haircut.id


def test_case_and_padding_do_not_matter(session: Session, haircut: Service) -> None:
    assert resolve_service(session, "  hAiRcUt ").id == haircut.id


def test_a_near_miss_is_not_guessed_at(session: Session, haircut: Service) -> None:
    """No fuzzy matching: "Haircuts" is not "Haircut"."""
    with pytest.raises(ServiceLookupError):
        resolve_service(session, "Haircuts")


def test_an_unknown_name_reports_what_is_offered(
    session: Session, haircut: Service
) -> None:
    with pytest.raises(ServiceLookupError) as caught:
        resolve_service(session, "Hovercraft")
    assert caught.value.offered == ["Haircut"]


def test_two_active_services_with_one_name_are_ambiguous(
    session: Session, haircut: Service
) -> None:
    session.add(Service(name="haircut", duration_minutes=45, staff_id="alex"))
    session.commit()

    with pytest.raises(ServiceLookupError) as caught:
        resolve_service(session, "Haircut")
    assert "ambiguous" in str(caught.value)


def test_an_inactive_namesake_does_not_make_a_name_ambiguous(
    session: Session, haircut: Service
) -> None:
    session.add(
        Service(name="Haircut", duration_minutes=45, staff_id="alex", active=False)
    )
    session.commit()

    assert resolve_service(session, "Haircut").id == haircut.id


def test_only_active_services_are_listed_as_offered(
    session: Session, haircut: Service
) -> None:
    session.add(Service(name="Beard trim", duration_minutes=15, staff_id="sam"))
    session.add(
        Service(name="Perm", duration_minutes=90, staff_id="sam", active=False)
    )
    session.commit()

    assert offered_service_names(session) == ["Beard trim", "Haircut"]


def test_an_empty_service_name_is_refused(session: Session) -> None:
    with pytest.raises(ToolArgumentError):
        resolve_service(session, "  ")


# --- the context -----------------------------------------------------------


def test_the_context_reports_the_configured_timezone(
    session: Session, calendar_settings
) -> None:
    context = ToolContext(session=session, settings=calendar_settings)
    assert context.timezone() == ZoneInfo("UTC")


def test_the_context_builds_a_calendar_on_the_same_session(
    session: Session, calendar_settings
) -> None:
    from app.calendar import CalendarService

    context = ToolContext(session=session, settings=calendar_settings)
    assert isinstance(context.calendar(), CalendarService)
