"""`check_availability` reports what the calendar computed, and nothing else."""

from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.models import Service
from app.tools import ToolContext, check_availability

MONDAY = "2026-03-02"


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 3, 2, hour, minute, tzinfo=UTC)


def test_slots_come_back_as_iso_strings_in_the_business_timezone(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = check_availability(tools, service_name="Haircut", day=MONDAY)

    assert result.success
    assert result.data["timezone"] == "UTC"
    assert result.data["day"] == MONDAY
    assert result.data["slots"][0] == {
        "starts_at": "2026-03-02T09:00:00+00:00",
        "ends_at": "2026-03-02T09:30:00+00:00",
    }


def test_the_slots_are_exactly_what_the_calendar_offers(
    tools: ToolContext, calendar, open_weekdays, haircut: Service
) -> None:
    """No filtering, no rounding, no additions of the tool's own."""
    expected = calendar.available_slots(haircut.id, date(2026, 3, 2))
    result = check_availability(tools, service_name="Haircut", day=MONDAY)

    assert len(result.data["slots"]) == len(expected)
    assert [slot["starts_at"] for slot in result.data["slots"]] == [
        slot.starts_at.isoformat() for slot in expected
    ]


def test_a_booked_slot_disappears(
    tools: ToolContext, calendar, open_weekdays, haircut: Service
) -> None:
    calendar.book(
        service_id=haircut.id,
        starts_at=_at(10),
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )

    result = check_availability(tools, service_name="Haircut", day=MONDAY)

    starts = [slot["starts_at"] for slot in result.data["slots"]]
    assert "2026-03-02T10:00:00+00:00" not in starts
    assert "2026-03-02T10:30:00+00:00" in starts


def test_a_closed_day_is_a_success_with_no_slots(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    """Nothing free is an answer, not a failure."""
    result = check_availability(tools, service_name="Haircut", day="2026-03-07")

    assert result.success
    assert result.data["slots"] == []
    assert result.error is None


def test_a_day_that_is_not_a_date_fails_clearly(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = check_availability(tools, service_name="Haircut", day="next Tuesday")

    assert not result.success
    assert "ISO date" in result.error


def test_an_unknown_service_says_what_is_offered(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = check_availability(tools, service_name="Hovercraft", day=MONDAY)

    assert not result.success
    assert "Hovercraft" in result.error
    assert result.data["services_offered"] == ["Haircut"]


def test_availability_honours_a_non_utc_business_timezone(
    session: Session, open_weekdays, haircut: Service
) -> None:
    """The same instants, reported where the business actually is."""
    from app.config import Settings

    context = ToolContext(
        session=session,
        settings=Settings(
            _env_file=None,
            business_timezone="Europe/London",
            slot_granularity_minutes=15,
        ),
    )

    result = check_availability(context, service_name="Haircut", day=MONDAY)

    assert result.data["timezone"] == "Europe/London"
    # March 2nd is still GMT, so London and UTC agree on the offset.
    assert result.data["slots"][0]["starts_at"] == "2026-03-02T09:00:00+00:00"


def test_a_date_object_is_accepted_as_well_as_a_string(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    result = check_availability(tools, service_name="Haircut", day=date(2026, 3, 2))

    assert result.success
    assert result.data["day"] == MONDAY


def test_an_inactive_service_is_not_offered(
    session: Session, tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    haircut.active = False
    session.commit()

    result = check_availability(tools, service_name="Haircut", day=MONDAY)

    assert not result.success
    assert result.data["services_offered"] == []
