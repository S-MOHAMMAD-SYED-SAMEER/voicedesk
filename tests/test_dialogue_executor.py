"""Dispatching a model's tool call — and refusing the ones it may not make."""

import json
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dialogue.executor import UNVERIFIED_SLOT, ToolExecutor
from app.models import Appointment, Service
from app.providers.llm import ToolUse
from app.tools import ToolContext

MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"
ELEVEN = "2026-03-02T11:00:00+00:00"


@pytest.fixture
def executor(tools: ToolContext) -> ToolExecutor:
    return ToolExecutor(tools)


def _use(name: str, arguments) -> ToolUse:
    return ToolUse(id=f"toolu_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)


def _run(executor: ToolExecutor, name: str, arguments):
    return executor.execute(_use(name, arguments))


def _check(executor: ToolExecutor, service: str = "Haircut", day: str = MONDAY):
    return _run(executor, "check_availability", {"service_name": service, "day": day})


def _book(executor: ToolExecutor, starts_at: str = TEN, service: str = "Haircut"):
    return _run(
        executor,
        "book_appointment",
        {
            "service_name": service,
            "starts_at": starts_at,
            "customer_name": "Ada Lovelace",
            "phone": "+447700900123",
        },
    )


# --- dispatch -------------------------------------------------------------


def test_a_tool_runs_through_the_registry(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    block, record = _check(executor)

    assert record.tool_name == "check_availability"
    assert record.success is True
    assert block["is_error"] is False
    assert json.loads(block["content"])["data"]["slots"]


def test_the_result_block_carries_the_tool_use_id(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """Without the id the model cannot match a result to the call it made."""
    use = _use("check_availability", {"service_name": "Haircut", "day": MONDAY})

    block, _ = executor.execute(use)

    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == use.id


def test_a_failure_is_a_result_not_an_exception(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    block, record = _check(executor, service="Hovercraft")

    assert record.success is False
    assert block["is_error"] is True
    assert "Hovercraft" in record.error


def test_recovery_data_reaches_the_model_on_a_failure(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """The model needs to know what is offered before it can ask again."""
    block, record = _check(executor, service="Hovercraft")

    payload = json.loads(block["content"])
    assert payload["data"]["services_offered"] == ["Haircut"]
    assert record.data["services_offered"] == ["Haircut"]


def test_an_unknown_tool_is_recorded_rather_than_raised(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """The wrong-tool-call rate is a metric, so the attempt has to survive."""
    block, record = _run(executor, "order_a_taxi", {})

    assert record.tool_name == "order_a_taxi"
    assert record.success is False
    assert block["is_error"] is True
    assert "not a tool" in record.error


def test_an_argument_the_tool_does_not_take_is_a_failure_not_a_crash(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    block, record = _run(
        executor, "cancel", {"appointment_id": str(uuid.uuid4()), "colour": "blue"}
    )

    assert record.success is False
    assert block["is_error"] is True
    assert "could not be called with those arguments" in record.error


def test_a_missing_argument_is_a_failure_not_a_crash(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    block, record = _run(executor, "cancel", {})

    assert record.success is False
    assert "could not be called with those arguments" in record.error


def test_arguments_that_are_not_an_object_are_refused(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    block, record = _run(executor, "cancel", "the ten o'clock one")

    assert record.success is False
    assert record.arguments == {}
    assert block["is_error"] is True


def test_malformed_argument_values_get_the_tools_own_message(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    _, record = _check(executor, day="next Tuesday")

    assert record.success is False
    assert "ISO date" in record.error


def test_the_arguments_are_recorded_verbatim(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    arguments = {"service_name": "  haircut ", "day": MONDAY}

    _, record = _run(executor, "check_availability", arguments)

    assert record.arguments == arguments


def test_latency_is_measured(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    _, record = _check(executor)

    assert record.latency_ms is not None and record.latency_ms >= 0


def test_the_block_content_is_the_whole_serialised_result(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    block, _ = _check(executor)

    payload = json.loads(block["content"])
    assert set(payload) == {"success", "data", "error"}


# --- the booking guard ----------------------------------------------------


def test_booking_without_checking_availability_is_blocked(
    session: Session, executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """Case 1: an arbitrary time the calendar never offered."""
    block, record = _book(executor)

    assert record.success is False
    assert record.error == UNVERIFIED_SLOT
    assert block["is_error"] is True
    assert json.loads(block["content"])["data"]["unverified_slot"] is True
    assert session.execute(select(Appointment)).first() is None


def test_booking_a_slot_that_was_offered_proceeds(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """Case 2: checked, offered 10:00, books 10:00."""
    _, check = _check(executor)
    assert TEN in [slot["starts_at"] for slot in check.data["slots"]]

    _, record = _book(executor, starts_at=TEN)

    assert record.success is True, record.error
    assert record.data["starts_at"] == TEN


def test_booking_a_time_that_was_not_among_the_offered_slots_is_blocked(
    session: Session, executor: ToolExecutor, haircut: Service
) -> None:
    """Case 3: offered 10:00 and nothing else, attempts 11:00.

    Saturday opens for half an hour, so the day has exactly one bookable
    start time and 11:00 is unambiguously a time the calendar never offered.
    """
    from datetime import time

    from app.models import BusinessHours

    session.add(BusinessHours(weekday=5, opens_at=time(10), closes_at=time(10, 30)))
    session.commit()

    _, check = _check(executor, day="2026-03-07")
    assert [slot["starts_at"] for slot in check.data["slots"]] == [
        "2026-03-07T10:00:00+00:00"
    ]

    _, record = _book(executor, starts_at="2026-03-07T11:00:00+00:00")

    assert record.success is False
    assert record.error == UNVERIFIED_SLOT
    assert session.execute(select(Appointment)).first() is None


def test_only_the_service_that_was_checked_is_verified(
    session: Session, executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """A slot offered for one service does not verify another."""
    session.add(Service(name="Beard trim", duration_minutes=15, staff_id="alex"))
    session.commit()
    _check(executor, service="Haircut")

    _, record = _book(executor, starts_at=TEN, service="Beard trim")

    assert record.success is False
    assert record.error == UNVERIFIED_SLOT


def test_a_failed_availability_check_verifies_nothing(
    session: Session, executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """Case 4: a check that failed cannot stand in for one that succeeded."""
    _, check = _check(executor, service="Hovercraft")
    assert check.success is False

    _, record = _book(executor, starts_at=TEN)

    assert record.success is False
    assert record.error == UNVERIFIED_SLOT
    assert session.execute(select(Appointment)).first() is None


def test_an_empty_day_verifies_nothing(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """Saturday is closed under `open_weekdays`, so nothing is offered."""
    _, check = _check(executor, day="2026-03-07")
    assert check.success and check.data["slots"] == []

    _, record = _book(executor, starts_at="2026-03-07T10:00:00+00:00")

    assert record.success is False
    assert record.error == UNVERIFIED_SLOT


def test_the_guard_compares_instants_not_strings(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """The same moment written a different way is the same moment."""
    _check(executor)

    _, record = _book(executor, starts_at="2026-03-02T05:00:00-05:00")

    assert record.success is True, record.error


def test_the_guard_ignores_case_and_padding_in_the_service_name(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    _check(executor)

    _, record = _book(executor, starts_at=TEN, service="  haircut ")

    assert record.success is True, record.error


def test_an_unparsable_time_reaches_the_tool_for_its_own_message(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """The tool's error is better than the guard's; do not shadow it."""
    _check(executor)

    _, record = _book(executor, starts_at="tomorrow morning")

    assert record.success is False
    assert "ISO timestamp" in record.error


def test_the_guard_does_not_stop_a_second_booking_of_an_offered_slot(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """Case 5: the database, not the guard, settles a slot that has gone."""
    _check(executor)
    first, _ = _book(executor, starts_at=TEN)

    _, record = _book(executor, starts_at=TEN)

    assert record.success is False
    assert record.data["slot_taken"] is True
    assert record.error != UNVERIFIED_SLOT


def test_reschedule_is_not_guarded(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    """Deliberate: the calendar refuses an unopen or taken time anyway."""
    _check(executor)
    _, booked = _book(executor, starts_at=TEN)

    _, record = _run(
        executor,
        "reschedule",
        {"appointment_id": booked.data["appointment_id"], "new_starts_at": ELEVEN},
    )

    assert record.success is True, record.error


def test_the_ledger_is_per_conversation(
    tools: ToolContext, open_weekdays, haircut: Service
) -> None:
    """One caller's verified slots cannot verify another caller's booking."""
    first = ToolExecutor(tools)
    _check(first)
    second = ToolExecutor(tools)

    _, record = _book(second, starts_at=TEN)

    assert record.success is False
    assert record.error == UNVERIFIED_SLOT
    assert first.offered_slots and not second.offered_slots


def test_offered_slots_are_recorded_as_instants(
    executor: ToolExecutor, open_weekdays, haircut: Service
) -> None:
    _check(executor)

    assert ("haircut", datetime(2026, 3, 2, 10, tzinfo=UTC)) in executor.offered_slots
