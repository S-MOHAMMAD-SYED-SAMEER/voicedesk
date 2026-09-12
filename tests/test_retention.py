"""Deleting old calls: only when asked, only what was asked for.

This is the one command in the repository that destroys data, so most of
these tests are about what it refuses to do. There is no default age, nothing
goes without `--confirm`, and a recent call is never touched.

The cascades are the other half. Turns, tool calls and cost rows go with the
call; appointments do not — a booking outlives the conversation that made it,
and deleting a transcript must not cancel somebody's haircut.
"""

from datetime import UTC, datetime, time, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.calendar import CalendarService
from app.config import Settings
from app.models import (
    Appointment,
    BusinessHours,
    Call,
    CallCost,
    CallDirection,
    CostComponent,
    Service,
    ToolCall,
    Turn,
    TurnRole,
)
from app.retention import MINIMUM_DAYS, RetentionError, cutoff, main, purge, survey

NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


def _call(session: Session, *, age_days: float, sid: str) -> Call:
    call = Call(
        direction=CallDirection.INBOUND,
        from_number="+447700900123",
        to_number="+441234567890",
        provider_call_sid=sid,
        started_at=datetime.now(UTC) - timedelta(days=age_days),
    )
    session.add(call)
    session.commit()
    return call


@pytest.fixture
def history(session: Session):
    """One old call with everything hanging off it, and one recent call."""
    session.add_all(
        [
            BusinessHours(weekday=day, opens_at=time(9), closes_at=time(17))
            for day in range(5)
        ]
    )
    service = Service(name="Haircut", duration_minutes=30, staff_id="sam")
    session.add(service)
    session.commit()

    old = _call(session, age_days=120, sid="CA-old")
    recent = _call(session, age_days=3, sid="CA-recent")

    caller = Turn(call_id=old.id, role=TurnRole.CALLER, text="Book me in.")
    reply = Turn(call_id=old.id, role=TurnRole.AGENT, text="Of course.")
    session.add_all([caller, reply])
    session.commit()
    session.add(
        ToolCall(
            turn_id=reply.id,
            tool_name="check_availability",
            arguments={"service_name": "Haircut"},
            result={"ok": True},
            success=True,
        )
    )
    session.add(
        CallCost(
            call_id=old.id,
            turn_id=reply.id,
            component=CostComponent.LLM,
            provider="a-provider",
            input_units=1000,
            unit_type="tokens",
        )
    )
    session.commit()

    calendar = CalendarService(
        session, Settings(_env_file=None, business_timezone="UTC")
    )
    appointment = calendar.book(
        service_id=service.id,
        starts_at=datetime(2026, 3, 2, 10, tzinfo=UTC),
        customer_name="Ada Lovelace",
        phone="+447700900123",
        call_id=old.id,
    )
    return old, recent, appointment


# --- the age boundary ------------------------------------------------------


def test_the_cutoff_is_that_many_days_back() -> None:
    assert cutoff(90, now=NOW) == NOW - timedelta(days=90)


def test_an_age_below_the_minimum_is_refused() -> None:
    with pytest.raises(RetentionError):
        cutoff(MINIMUM_DAYS - 1, now=NOW)


def test_zero_days_is_refused() -> None:
    """Otherwise a stray `0` deletes calls that are still in progress."""
    with pytest.raises(RetentionError, match="in progress"):
        cutoff(0, now=NOW)


def test_a_negative_age_is_refused() -> None:
    with pytest.raises(RetentionError):
        cutoff(-30, now=NOW)


def test_there_is_no_default_age() -> None:
    """`--older-than` is required, so nothing can run by accident."""
    with pytest.raises(SystemExit):
        main([])


# --- surveying -------------------------------------------------------------


def test_the_survey_counts_what_would_go(session: Session, history) -> None:
    counts = survey(session, cutoff(90))

    assert counts["calls"] == 1
    assert counts["turns"] == 2
    assert counts["tool_calls"] == 1
    assert counts["call_costs"] == 1


def test_the_survey_counts_appointments_separately(session: Session, history) -> None:
    """Reported because they stay, not because they go."""
    assert survey(session, cutoff(90))["appointments_detached"] == 1


def test_the_survey_deletes_nothing(session: Session, history) -> None:
    survey(session, cutoff(90))

    assert len(session.execute(select(Call)).scalars().all()) == 2


def test_a_cutoff_older_than_everything_finds_nothing(
    session: Session, history
) -> None:
    assert survey(session, cutoff(3650))["calls"] == 0


# --- deleting --------------------------------------------------------------


def test_the_old_call_goes(session: Session, history) -> None:
    old, _recent, _appointment = history

    assert purge(session, cutoff(90)) == 1
    assert session.get(Call, old.id) is None


def test_the_recent_call_stays(session: Session, history) -> None:
    """The whole point of an age."""
    _old, recent, _appointment = history
    purge(session, cutoff(90))

    assert session.get(Call, recent.id) is not None


def test_the_transcript_goes_with_the_call(session: Session, history) -> None:
    purge(session, cutoff(90))

    assert session.execute(select(Turn)).scalars().all() == []


def test_the_tool_calls_go_with_the_turns(session: Session, history) -> None:
    """Two cascades deep. The arguments held a name and a number."""
    purge(session, cutoff(90))

    assert session.execute(select(ToolCall)).scalars().all() == []


def test_the_cost_rows_go_with_the_call(session: Session, history) -> None:
    purge(session, cutoff(90))

    assert session.execute(select(CallCost)).scalars().all() == []


def test_the_appointment_survives_its_call(session: Session, history) -> None:
    """Deleting a transcript must not cancel somebody's haircut."""
    _old, _recent, appointment = history
    purge(session, cutoff(90))

    kept = session.get(Appointment, appointment.id)
    assert kept is not None
    assert kept.call_id is None


def test_deleting_nothing_is_not_an_error(session: Session, history) -> None:
    assert purge(session, cutoff(3650)) == 0
    assert len(session.execute(select(Call)).scalars().all()) == 2


# --- the command line ------------------------------------------------------


@pytest.fixture
def command(session: Session, monkeypatch: pytest.MonkeyPatch):
    """Run `main` against the test database, on this session."""
    from contextlib import contextmanager

    @contextmanager
    def borrowed():
        yield session

    monkeypatch.setattr("app.retention.get_sessionmaker", lambda: borrowed)
    return main


def test_a_report_is_printed_and_nothing_is_deleted(
    command, session: Session, history, capsys
) -> None:
    assert command(["--older-than", "90"]) == 0

    assert "Nothing was deleted" in capsys.readouterr().out
    assert len(session.execute(select(Call)).scalars().all()) == 2


def test_dry_run_deletes_nothing_either(
    command, session: Session, history, capsys
) -> None:
    assert command(["--older-than", "90", "--dry-run"]) == 0

    assert len(session.execute(select(Call)).scalars().all()) == 2


def test_confirm_is_what_deletes(command, session: Session, history, capsys) -> None:
    assert command(["--older-than", "90", "--confirm"]) == 0

    assert "Deleted 1 call(s)." in capsys.readouterr().out
    assert len(session.execute(select(Call)).scalars().all()) == 1


def test_an_age_below_the_minimum_exits_without_deleting(
    command, session: Session, history, capsys
) -> None:
    assert command(["--older-than", "0", "--confirm"]) == 2

    assert len(session.execute(select(Call)).scalars().all()) == 2


def test_the_report_names_nobody(command, session: Session, history, capsys) -> None:
    """A retention report that quoted what it deleted would be a copy of it."""
    command(["--older-than", "90", "--confirm"])
    written = capsys.readouterr().out

    assert "Ada Lovelace" not in written
    assert "+447700900123" not in written
    assert "Book me in." not in written


# --- what the command cannot do --------------------------------------------


def test_nothing_here_can_empty_a_table() -> None:
    """No `TRUNCATE`, and no bulk delete that could run without a filter.

    Every deletion goes through `session.delete(call)` on one row at a time,
    which is also what makes the schema's cascades run. A `delete()`
    *statement* would skip them, and a mistake in its `where` clause would
    empty a table.
    """
    import ast
    import pathlib

    import app.retention as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "sqlalchemy"
        for alias in node.names
    }
    assert "delete" not in imported
    assert "text" not in imported

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "delete":
                assert isinstance(node.func.value, ast.Name)
                assert node.func.value.id == "session"

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "truncate" not in node.value.lower() or node.value.startswith(
                "Deleting calls"
            )
