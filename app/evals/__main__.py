"""`python -m app.evals` — run the suite and say what happened.

    python -m app.evals                       every scenario
    python -m app.evals --list                names only, runs nothing
    python -m app.evals --scenario booking_available --scenario take_message
    python -m app.evals --json report.json    machine-readable, as well as text
    python -m app.evals --fail-under 85       exit 1 below that task success
    python -m app.evals --quiet               the headline only

Shaped after `python -m app.metrics`, which is the other reporting entry point
in this repository. There is no HTTP endpoint and no dashboard: an evaluation
suite is something you run, read, and check into a case study.

Exit codes: 0 when the suite ran and met its threshold, 1 when it did not, and
2 when it could not run at all — a database it is not allowed to touch, or a
dataset that does not load.
"""

import argparse
import pathlib
import sys
from collections.abc import Sequence

from app.evals import EVAL_SUITE_VERSION, dataset
from app.evals.checks import assess
from app.evals.database import EvalDatabaseError
from app.evals.report import as_json, render
from app.evals.runner import EvalRunner
from app.evals.scenario import Scenario, ScenarioError
from app.evals.scoring import Report, ScenarioResult, score

CANNOT_RUN = 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)

    try:
        scenarios = _selected(arguments.scenario)
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CANNOT_RUN

    if arguments.list:
        for scenario in scenarios:
            print(f"{scenario.name}\t{scenario.kind}\t{scenario.description}")
        return 0

    try:
        report = run(scenarios)
    except EvalDatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CANNOT_RUN

    if arguments.json:
        pathlib.Path(arguments.json).write_text(as_json(report), encoding="utf-8")

    print(_output(report, quiet=arguments.quiet))

    return _exit_code(report, arguments.fail_under)


def run(scenarios: Sequence[Scenario]) -> Report:
    """Every scenario, against one migrated evaluation database."""
    results: list[ScenarioResult] = []
    with EvalRunner() as runner:
        for scenario in scenarios:
            trace = runner.run(scenario)
            results.append(
                ScenarioResult(
                    scenario=scenario,
                    trace=trace,
                    assessment=assess(scenario, trace),
                )
            )
    return score(results, EVAL_SUITE_VERSION)


# --- internals -------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evals",
        description=(
            "Run the deterministic VoiceDesk evaluation suite against an "
            "isolated evaluation database. No provider is called and no "
            "credential is needed."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the scenarios and run nothing.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        metavar="NAME",
        help="Run only this scenario. May be repeated.",
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Also write the report as JSON to this path.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="PERCENT",
        help="Exit 1 if task success is below this percentage.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print the headline numbers only.",
    )
    return parser


def _selected(names: list[str] | None) -> tuple[Scenario, ...]:
    if not names:
        return dataset.all_scenarios()
    return dataset.by_name(tuple(names))


def _output(report: Report, *, quiet: bool) -> str:
    text = render(report)
    if not quiet:
        return text
    # Everything above the per-scenario detail, which the separator marks.
    head, _, _ = text.partition("-" * 62)
    return head.rstrip()


def _exit_code(report: Report, threshold: float | None) -> int:
    if threshold is None:
        return 0 if report.failed == 0 else 1
    percent = report.task_success.percent
    if percent is None:
        return 1
    return 0 if percent >= threshold else 1


if __name__ == "__main__":  # pragma: no cover - the entry point itself
    raise SystemExit(main())
