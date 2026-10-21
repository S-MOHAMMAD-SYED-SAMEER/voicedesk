"""The model is told the date, time and time zone on every turn, and the system refuses the past.

The real model's reading of "tomorrow" is for the evaluation to judge. What is proved here is everything the system
controls: that the facts reach the model in words it cannot misread, that they are right for the clock and the
business time zone, and that a past time cannot be booked whatever the model asks for.
"""

import re
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from app.clock import fixed_clock
from app.config import Settings
from app.dialogue import Conversation
from app.dialogue.prompt import DAYS_AHEAD, clock_context
from app.providers.llm import ModelResponse
from app.tools import ToolContext, book_appointment, check_availability
from tests.conftest import FakeModel, say, use_tools

SUNDAY_9AM = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
WEDNESDAY_NOON = datetime(2026, 3, 4, 12, 0, tzinfo=UTC)


def _conversation(session, call, model, settings, now):
    return Conversation(session, call, model, settings, clock=fixed_clock(now))


class _ReadsTheDateFromThePrompt:
    """A stand-in for a model that does exactly what the prompt says: use the dates as written."""

    def __init__(self, picker, service="Haircut") -> None:
        self._picker = picker
        self._service = service
        self._asked = False
        self.systems: list[str] = []

    def respond(self, *, system, messages, tools) -> ModelResponse:
        self.systems.append(system)
        if not self._asked:
            self._asked = True
            return use_tools(
                (
                    "check_availability",
                    {"service_name": self._service, "day": self._picker(system)},
                )
            )
        return say("Here is what is free.")


def _tomorrow(system: str) -> str:
    return re.search(r"tomorrow is (\d{4}-\d{2}-\d{2})", system).group(1)


def _next_monday(system: str) -> str:
    return re.search(r"Mon (\d{4}-\d{2}-\d{2})", system).group(1)


# --- what the model is told ------------------------------------------------


def test_the_prompt_names_today_tomorrow_and_the_zone() -> None:
    text = clock_context(SUNDAY_9AM, ZoneInfo("UTC"))

    assert "It is Sunday 2026-03-01, 09:00 in UTC (UTC+00:00)" in text
    assert "tomorrow is 2026-03-02 (Monday)" in text


def test_the_prompt_uses_the_business_time_zone_not_utc() -> None:
    # 23:30 UTC on 21 October is 00:30 on the 22nd in London (British Summer Time, UTC+01:00).
    text = clock_context(datetime(2026, 10, 21, 23, 30, tzinfo=UTC), ZoneInfo("Europe/London"))

    assert "Thursday 2026-10-22, 00:30 in Europe/London (UTC+01:00)" in text
    assert "tomorrow is 2026-10-23 (Friday)" in text


def test_the_prompt_lists_the_days_ahead_with_their_weekdays() -> None:
    text = clock_context(WEDNESDAY_NOON, ZoneInfo("UTC"))

    assert "Thu 2026-03-05" in text and "Mon 2026-03-09" in text
    assert len(re.findall(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2}", text)) == DAYS_AHEAD


def test_every_model_request_carries_the_clock_after_the_constant_prompt(
    session, call, open_weekdays, haircut, calendar_settings
) -> None:
    model = FakeModel(say("Hello."), say("Still here."))
    conversation = _conversation(session, call, model, calendar_settings, SUNDAY_9AM)

    conversation.send("Hi")
    conversation.send("Hello again")

    for request in model.requests:
        assert request["system"].startswith(conversation.system_prompt)
        assert "# Right now" in request["system"]
        assert "Today is 2026-03-01" in request["system"]


def test_the_clock_is_read_each_turn_not_once_per_call(
    session, call, open_weekdays, haircut, calendar_settings
) -> None:
    now = [SUNDAY_9AM]
    model = FakeModel(say("One."), say("Two."))
    conversation = Conversation(session, call, model, calendar_settings, clock=lambda: now[0])

    conversation.send("first")
    now[0] = datetime(2026, 3, 3, 9, 0, tzinfo=UTC)
    conversation.send("second")

    assert "Today is 2026-03-01" in model.requests[0]["system"]
    assert "Today is 2026-03-03" in model.requests[1]["system"]


def test_the_clock_line_is_not_part_of_what_the_caller_said(
    session, call, open_weekdays, haircut, calendar_settings
) -> None:
    from sqlalchemy import select

    from app.models import Turn

    model = FakeModel(say("Hello."))
    _conversation(session, call, model, calendar_settings, SUNDAY_9AM).send("Hi")

    texts = [row.text for row in session.execute(select(Turn)).scalars()]

    assert not any("Right now" in text for text in texts)


# --- "tomorrow" and "next Monday", as the prompt lets a model resolve them --


def test_tomorrow_resolves_to_the_next_day_and_has_slots(
    session, call, open_weekdays, haircut, calendar_settings
) -> None:
    model = _ReadsTheDateFromThePrompt(_tomorrow)

    result = _conversation(session, call, model, calendar_settings, SUNDAY_9AM).send(
        "Can I come in tomorrow?"
    )

    record = result.tool_calls[0]
    assert record.success and record.data["day"] == "2026-03-02"
    assert record.data["slots"], "Monday is open"


def test_next_monday_resolves_from_the_list_of_days_ahead(
    session, call, open_weekdays, haircut, calendar_settings
) -> None:
    model = _ReadsTheDateFromThePrompt(_next_monday)

    result = _conversation(session, call, model, calendar_settings, WEDNESDAY_NOON).send(
        "Next Monday, please."
    )

    record = result.tool_calls[0]
    assert record.data["day"] == "2026-03-09"
    assert date.fromisoformat(record.data["day"]).weekday() == 0
    assert record.data["slots"]


def test_tomorrow_is_computed_in_the_business_time_zone(
    session, call, open_weekdays, haircut
) -> None:
    settings = Settings(_env_file=None, business_timezone="Pacific/Auckland")
    # 20:00 UTC on Sunday 1 March is 09:00 on Monday 2 March in Auckland (NZDT, UTC+13).
    now = datetime(2026, 3, 1, 20, 0, tzinfo=UTC)
    model = _ReadsTheDateFromThePrompt(_tomorrow)

    result = _conversation(session, call, model, settings, now).send("tomorrow?")

    assert result.tool_calls[0].data["day"] == "2026-03-03"


# --- a time that has passed -----------------------------------------------


def test_asking_about_a_day_gone_by_offers_nothing(
    session, call, open_weekdays, haircut, calendar_settings
) -> None:
    context = ToolContext(
        session=session, settings=calendar_settings, clock=fixed_clock(WEDNESDAY_NOON)
    )

    result = check_availability(context, service_name="Haircut", day="2026-03-02")

    assert result.success and result.data["slots"] == []


def test_a_past_time_is_refused_by_the_booking_tool_whatever_the_model_asked(
    session, open_weekdays, haircut, calendar_settings
) -> None:
    context = ToolContext(
        session=session, settings=calendar_settings, clock=fixed_clock(WEDNESDAY_NOON)
    )

    result = book_appointment(
        context,
        service_name="Haircut",
        starts_at="2026-03-02T10:00:00+00:00",
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )

    assert not result.success
    assert "already passed" in result.error


def test_a_past_time_earlier_today_is_refused_too(
    session, open_weekdays, haircut, calendar_settings
) -> None:
    context = ToolContext(
        session=session, settings=calendar_settings, clock=fixed_clock(WEDNESDAY_NOON)
    )

    result = book_appointment(
        context,
        service_name="Haircut",
        starts_at="2026-03-04T09:00:00+00:00",
        customer_name="Ada Lovelace",
        phone="+447700900123",
    )

    assert not result.success and "already passed" in result.error


def test_a_past_time_cannot_be_booked_through_a_conversation_either(
    session, call, open_weekdays, haircut, calendar_settings
) -> None:
    """A model that offers a past time it was never given is stopped twice: the executor, then the calendar."""
    model = FakeModel(
        use_tools(
            (
                "book_appointment",
                {
                    "service_name": "Haircut",
                    "starts_at": "2026-03-02T10:00:00+00:00",
                    "customer_name": "Ada Lovelace",
                    "phone": "+447700900123",
                },
            )
        ),
        say("I'm afraid that time has passed."),
    )

    result = _conversation(session, call, model, calendar_settings, WEDNESDAY_NOON).send(
        "Book me for Monday at ten."
    )

    assert result.booked_appointment_id is None
    assert not result.tool_calls[0].success
