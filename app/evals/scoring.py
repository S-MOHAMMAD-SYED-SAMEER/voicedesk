"""Adding up 18 calls into the numbers the specification asks for.

Every ratio here is reported with its denominator, and every denominator of
zero is reported as `N/A` rather than as `0%`. With eighteen scenarios one
failure moves task success by 5.6 points and a p95 is arithmetic rather than
evidence; a bare percentage would invite a confidence nobody has earned.

Two things are counted apart from task success, on purpose:

* **unverified reschedules**, which are an observation about frozen
  milestone-3/4 behaviour rather than a failure of it;
* **scenario-authoring failures**, which are the dataset's fault, and which
  would be a lie about the receptionist if they were folded into its score.

Both still appear in the report. Not counting something against a score is not
the same as hiding it.
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.evals.checks import (
    AUTHORING,
    HALLUCINATED_AVAILABILITY,
    UNVERIFIED_RESCHEDULE,
    Assessment,
    ToolTally,
)
from app.evals.scenario import Scenario
from app.evals.trace import CallTrace

# Below this many samples a percentile is arithmetic rather than evidence.
# The same threshold `app/metrics.py` uses, for the same reason.
MEANINGFUL_SAMPLE = 20


@dataclass(frozen=True)
class Ratio:
    """A fraction that always shows its denominator."""

    numerator: int
    denominator: int

    @property
    def defined(self) -> bool:
        return self.denominator > 0

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.defined else None

    @property
    def percent(self) -> float | None:
        value = self.value
        return None if value is None else value * 100

    def __str__(self) -> str:
        if not self.defined:
            return "N/A (0 of 0)"
        return f"{self.percent:.1f}% ({self.numerator}/{self.denominator})"


@dataclass(frozen=True)
class Percentiles:
    """A distribution, with the honesty attached."""

    count: int
    p50: float | None
    p95: float | None

    @property
    def meaningful(self) -> bool:
        return self.count >= MEANINGFUL_SAMPLE


@dataclass
class ScenarioResult:
    """One scenario's verdict, with everything needed to explain it."""

    scenario: Scenario
    trace: CallTrace
    assessment: Assessment

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def passed(self) -> bool:
        return self.assessment.passed

    @property
    def authoring_fault(self) -> bool:
        """Did this fail because of the dataset rather than the system?"""
        return any(finding.authoring for finding in self.assessment.blocking)

    @property
    def tools_taken(self) -> str:
        return (
            " → ".join(
                f"{record.tool_name}{'' if record.success else '(failed)'}"
                for record in self.trace.tool_calls
            )
            or "none"
        )


@dataclass
class Report:
    """Everything the suite measured."""

    version: str
    results: list[ScenarioResult] = field(default_factory=list)
    tools: ToolTally = field(default_factory=ToolTally)

    # --- headline ----------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def task_success(self) -> Ratio:
        return Ratio(self.passed, self.total)

    @property
    def authoring_failures(self) -> int:
        return sum(1 for result in self.results if result.authoring_fault)

    # --- the specification's numbers ---------------------------------------

    @property
    def hallucinated_availability(self) -> int:
        return self._count(HALLUCINATED_AVAILABILITY)

    @property
    def unverified_reschedules(self) -> int:
        return self._count(UNVERIFIED_RESCHEDULE)

    @property
    def tool_correctness(self) -> Ratio:
        return Ratio(self.tools.correct, self.tools.total)

    @property
    def escalation_precision(self) -> Ratio:
        """Of the calls that escalated, how many should have?"""
        escalated = [
            result for result in self.results if result.trace.escalated
        ]
        correct = sum(
            1 for result in escalated if result.scenario.expect.should_escalate
        )
        return Ratio(correct, len(escalated))

    @property
    def escalation_recall(self) -> Ratio:
        """Of the calls that should have escalated, how many did?"""
        required = [
            result
            for result in self.results
            if result.scenario.expect.should_escalate
        ]
        correct = sum(1 for result in required if result.trace.escalated)
        return Ratio(correct, len(required))

    @property
    def turns_to_booking(self) -> Percentiles:
        return percentiles(
            [
                float(turn)
                for result in self.results
                if (turn := result.trace.booked_turn) is not None
            ]
        )

    # --- cost, from milestone 8 --------------------------------------------

    @property
    def measured_calls(self) -> int:
        return sum(
            1 for result in self.results if result.trace.total_cost_usd is not None
        )

    @property
    def unpriced_calls(self) -> int:
        return self.total - self.measured_calls

    @property
    def total_cost_usd(self):
        """The sum of the calls that could be priced, or `None` if none could."""
        priced = [
            result.trace.total_cost_usd
            for result in self.results
            if result.trace.total_cost_usd is not None
        ]
        return sum(priced) if priced else None

    # --- findings ----------------------------------------------------------

    @property
    def findings_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            for finding in result.assessment.findings:
                counts[finding.category] = counts.get(finding.category, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def behavioural_failures(self) -> dict[str, int]:
        """Failure categories that are VoiceDesk's, not the dataset's."""
        return {
            category: count
            for category, count in self.findings_by_category.items()
            if category not in AUTHORING and category != UNVERIFIED_RESCHEDULE
        }

    def _count(self, category: str) -> int:
        return sum(
            1
            for result in self.results
            for finding in result.assessment.findings
            if finding.category == category
        )


def score(results: Sequence[ScenarioResult], version: str) -> Report:
    """Every scenario's verdict, added up."""
    report = Report(version=version, results=list(results))
    for result in results:
        report.tools += result.assessment.tools
    return report


def percentiles(values: Sequence[float]) -> Percentiles:
    """The median and the 95th, or nothing when there is nothing to say.

    One sample is its own median and its own p95, which is worth reporting and
    not worth interpolating. The same rule `app/metrics.py` follows.
    """
    ordered = sorted(values)
    if not ordered:
        return Percentiles(0, None, None)
    if len(ordered) == 1:
        return Percentiles(1, ordered[0], ordered[0])

    cuts = statistics.quantiles(ordered, n=100, method="inclusive")
    return Percentiles(len(ordered), statistics.median(ordered), cuts[94])


__all__ = [
    "MEANINGFUL_SAMPLE",
    "Percentiles",
    "Ratio",
    "Report",
    "ScenarioResult",
    "percentiles",
    "score",
]
