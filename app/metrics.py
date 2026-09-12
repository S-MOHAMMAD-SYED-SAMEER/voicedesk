"""How quickly the receptionist answers, reported honestly.

    python -m app.metrics

Two numbers, both defined against the moment the caller stopped talking,
because that is the moment they start waiting:

* **first_audio_latency_ms** — until the first audio of the reply exists.
  This is what the specification's 1.2-second budget is about.
* **turn_latency_ms** — until the whole reply has been synthesised.

Both are reconstructed from the columns `turns` already has, so there is no
metrics table, no time series and no new migration. Rows written before
realtime existed have null latencies and are skipped rather than counted as
zero; the count is printed beside every percentile so a p95 over four turns
is visibly a p95 over four turns.

This measures. It does not promise: whether the budget is met depends on
provider round-trips that this repository has never made.
"""

import argparse
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_sessionmaker
from app.models import Turn, TurnRole

# Below this many samples a percentile is arithmetic rather than evidence.
MEANINGFUL_SAMPLE = 20


@dataclass(frozen=True)
class Summary:
    """One metric's distribution, with the honesty attached."""

    name: str
    count: int
    p50: float | None
    p95: float | None

    @property
    def meaningful(self) -> bool:
        return self.count >= MEANINGFUL_SAMPLE


def percentiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    """The median and the 95th, or nothing when there is nothing to say.

    `statistics.quantiles` needs at least two points; one sample is its own
    median and its own p95, which is worth reporting and not worth
    interpolating.
    """
    ordered = sorted(values)
    if not ordered:
        return None, None
    if len(ordered) == 1:
        return ordered[0], ordered[0]

    cuts = statistics.quantiles(ordered, n=100, method="inclusive")
    return statistics.median(ordered), cuts[94]


def summarise(name: str, values: Sequence[float]) -> Summary:
    p50, p95 = percentiles(values)
    return Summary(name=name, count=len(values), p50=p50, p95=p95)


def turns_with_role(session: Session, role: TurnRole, call_id=None) -> list[Turn]:
    """Every turn of one role, oldest first.

    The two metrics live on different rows, because the numbers belong to
    different things: how long somebody spoke is a property of their own turn,
    and how long the answer took is a property of the answer.
    """
    query = select(Turn).where(Turn.role == role)
    if call_id is not None:
        query = query.where(Turn.call_id == call_id)
    return list(session.execute(query.order_by(Turn.created_at)).scalars().all())


def first_audio_latency_ms(turn: Turn) -> int | None:
    """Caller stops → first audio of the reply exists.

    Recognition, then the model and its tools, then synthesis up to the first
    chunk. Null on any turn missing a component, because a partial sum would
    read as a fast turn.
    """
    parts = (turn.stt_latency_ms, turn.llm_latency_ms, turn.tts_latency_ms)
    if any(part is None for part in parts):
        return None
    return sum(parts)


def collect(session: Session, call_id=None) -> dict[str, list[float]]:
    """Every measurable turn, as the two metrics."""
    first_audio = [
        latency
        for turn in turns_with_role(session, TurnRole.AGENT, call_id)
        if (latency := first_audio_latency_ms(turn)) is not None
    ]
    spoken = [
        turn.audio_ms
        for turn in turns_with_role(session, TurnRole.CALLER, call_id)
        if turn.audio_ms is not None
    ]
    return {"first_audio_latency_ms": first_audio, "caller_audio_ms": spoken}


def render(summaries: Sequence[Summary]) -> str:
    """The table, plus a line saying when not to believe it."""
    lines = [
        f"{'metric':<28}{'count':>8}{'p50':>10}{'p95':>10}",
        "-" * 56,
    ]
    thin = False
    for summary in summaries:
        p50 = "n/a" if summary.p50 is None else f"{summary.p50:.0f}"
        p95 = "n/a" if summary.p95 is None else f"{summary.p95:.0f}"
        mark = "" if summary.meaningful or not summary.count else " *"
        thin = thin or bool(mark)
        lines.append(f"{summary.name:<28}{summary.count:>8}{p50:>10}{p95:>10}{mark}")

    if thin:
        lines.append("")
        lines.append(
            f"* fewer than {MEANINGFUL_SAMPLE} samples: these are arithmetic, "
            "not evidence."
        )
    if not any(summary.count for summary in summaries):
        lines.append("")
        lines.append(
            "No turn carries latency yet. Those columns are written by the "
            "realtime path; turns from the utterance path leave them null."
        )
    return "\n".join(lines)


def report(session: Session, call_id=None) -> str:
    measured = collect(session, call_id)
    return render([summarise(name, values) for name, values in measured.items()])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.metrics",
        description="Turn latency across recorded calls.",
    )
    parser.add_argument("--call", help="Limit to one call id.", default=None)
    arguments = parser.parse_args(argv)

    with get_sessionmaker()() as session:
        print(report(session, arguments.call))
    return 0


if __name__ == "__main__":  # pragma: no cover - the entry point itself
    raise SystemExit(main())
