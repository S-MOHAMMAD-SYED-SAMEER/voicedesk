"""Deleting calls that are older than somebody decided they should be.

    python -m app.retention --older-than 90 --dry-run
    python -m app.retention --older-than 90 --confirm

VoiceDesk keeps full transcripts, caller names and telephone numbers, because
the specification asks it to: "Store every transcript. Redact nothing in v1,
but keep PII fields in named columns so redaction is easy to add later." What
it does not do is decide how long. **There is no default retention period and
this command invents none** — how long a business may keep a recording of a
customer is a legal and commercial question about that business, not
something a tool should answer on its behalf.

So the age is always explicit, and so is the deletion:

* `--older-than` has no default. Without it the command refuses to run.
* Nothing is deleted without `--confirm`. The default is a report.
* Deletion is by call, never by table. There is no `TRUNCATE` here and no
  statement that could empty anything: rows go one call at a time, and the
  foreign keys already decide what goes with them.

What goes with a call, by the schema's own cascades: its `turns`, their
`tool_calls`, and its `call_costs`. What does **not**: its `appointments`.
`appointments.call_id` is `ON DELETE SET NULL` because a booking outlives the
conversation that made it — deleting a transcript must not cancel somebody's
haircut. An appointment keeps its own `customer_name` and `phone`, so erasing
a caller completely is more than this command does, and the README says so.

Counts are logged. Names, numbers and transcripts are not: a retention run
that printed what it deleted would be a copy of the thing being deleted.
"""

import argparse
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_sessionmaker
from app.logging import configure as configure_logging
from app.models import Appointment, Call, CallCost, ToolCall, Turn

logger = logging.getLogger(__name__)

# Refusing this is the difference between a retention policy and an accident.
MINIMUM_DAYS = 1


class RetentionError(ValueError):
    """The command was asked to do something it will not do."""


def cutoff(days: int, *, now: datetime | None = None) -> datetime:
    """The instant before which a call counts as old."""
    if days < MINIMUM_DAYS:
        raise RetentionError(
            f"--older-than must be at least {MINIMUM_DAYS} day; {days} would "
            "delete calls that may still be in progress."
        )
    return (now or datetime.now(UTC)) - timedelta(days=days)


def survey(session: Session, before: datetime) -> dict[str, int]:
    """What would go, counted, without anything going.

    Counted by joining from the calls being deleted rather than by counting
    whole tables, so the numbers describe this deletion and not the database.
    """
    old = select(Call.id).where(Call.started_at < before).scalar_subquery()
    turns = select(Turn.id).where(Turn.call_id.in_(old)).scalar_subquery()

    return {
        "calls": _count(session, select(func.count()).select_from(Call).where(
            Call.started_at < before
        )),
        "turns": _count(session, select(func.count()).select_from(Turn).where(
            Turn.call_id.in_(old)
        )),
        "tool_calls": _count(session, select(func.count()).select_from(ToolCall).where(
            ToolCall.turn_id.in_(turns)
        )),
        "call_costs": _count(session, select(func.count()).select_from(CallCost).where(
            CallCost.call_id.in_(old)
        )),
        # Kept, not deleted. Reported so nobody is surprised by what stays.
        "appointments_detached": _count(
            session,
            select(func.count()).select_from(Appointment).where(
                Appointment.call_id.in_(old)
            ),
        ),
    }


def purge(session: Session, before: datetime) -> int:
    """Delete every call older than `before`, and return how many.

    One call at a time through the ORM, so the cascades the schema declares
    are the cascades that run. A bulk delete would skip them.
    """
    calls = (
        session.execute(select(Call).where(Call.started_at < before)).scalars().all()
    )
    for call in calls:
        session.delete(call)
    session.commit()
    return len(calls)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    configure_logging()

    try:
        before = cutoff(arguments.older_than)
    except RetentionError as exc:
        print(f"error: {exc}")
        return 2

    with get_sessionmaker()() as session:
        counts = survey(session, before)
        print(_report(before, counts, confirmed=arguments.confirm))

        if not arguments.confirm:
            return 0
        if not counts["calls"]:
            return 0

        deleted = purge(session, before)

    logger.info(
        "Retention: deleted %d call(s) started before %s.",
        deleted,
        before.isoformat(),
    )
    print(f"\nDeleted {deleted} call(s).")
    return 0


# --- internals -------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.retention",
        description=(
            "Delete calls older than a given age, with their transcripts, "
            "tool calls and cost rows. Reports by default; deletes only with "
            "--confirm. There is no default age: how long to keep a call is "
            "the deployer's decision, not this tool's."
        ),
    )
    parser.add_argument(
        "--older-than",
        type=int,
        required=True,
        metavar="DAYS",
        help="Delete calls that started more than this many days ago.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without it this only reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report and delete nothing. The default; accepted for clarity.",
    )
    return parser


def _count(session: Session, statement) -> int:
    return int(session.execute(statement).scalar_one() or 0)


def _report(before: datetime, counts: dict[str, int], *, confirmed: bool) -> str:
    lines = [
        f"Calls started before {before.isoformat()}",
        "",
        f"  calls          {counts['calls']}",
        f"  turns          {counts['turns']}",
        f"  tool_calls     {counts['tool_calls']}",
        f"  call_costs     {counts['call_costs']}",
        "",
        f"  appointments   {counts['appointments_detached']} kept, and detached "
        "from the deleted call",
    ]
    if not confirmed:
        lines += ["", "Nothing was deleted. Re-run with --confirm to delete."]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - the entry point itself
    raise SystemExit(main())
