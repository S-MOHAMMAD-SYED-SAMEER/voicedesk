"""`python -m realeval` — run the scenarios against a model.

    python -m realeval --list
    python -m realeval                                   # offline stub: verifies the machinery, measures nothing
    python -m realeval --provider anthropic --request-cap 200 --max-cost-usd 3.00 \\
        --input-usd-per-mtok 1 --output-usd-per-mtok 5 --out docs/results

A real run refuses to start unless **all** of this holds: both caps and both prices are given; adapter retries are off
(`VOICEDESK_DIALOGUE_MAX_RETRIES=0`, the default); a key is present (only the *names* are ever reported); and the
scenarios and rules match the digest recorded in `docs/EVAL.md` (they are frozen before any run). The caps are kept in a
ledger on disk and are shared by every run that uses it. The first 401, 403, 429, 503, 529, timeout, quota or credit
error stops the run: nothing is retried and nothing resumes without a person. A partial run is not a result and is not
written to `docs/results`.

Exit codes: 0 the run completed (whatever it found), 2 it could not start, 3 it was stopped.
"""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.config import Settings, get_settings
from app.evals.database import EvalDatabaseError
from app.evals.scoring import ScenarioResult, score
from realeval import REAL_EVAL_VERSION, freeze, results, scenarios
from realeval.budget import (
    DEFAULT_LEDGER,
    BudgetedModel,
    Prices,
    RunBudget,
    RunStopped,
    latency_summary,
)
from realeval.runner import RealEvalRunner, RealResult
from realeval.stub import StubModel

CANNOT_RUN = 2
STOPPED = 3
ROOT = Path(__file__).resolve().parent.parent
EVAL_DOC = ROOT / "docs" / "EVAL.md"
#: The model timeout for an evaluation run. The application's own limit (15 s, `Settings.dialogue_timeout_seconds`) is
#: for a caller on a telephone and is untouched; this one is for the evaluation only.
EVAL_TIMEOUT_SECONDS = 60.0
KEY_NAMES = ("VOICEDESK_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")


class CannotRun(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        selected = scenarios.by_name(tuple(arguments.scenario)) if arguments.scenario else scenarios.all_scenarios()
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return CANNOT_RUN

    if arguments.list:
        for item in selected:
            print(f"{item.name}\t{len(item.scenario.script)} turn(s)\t{item.scenario.description}")
        return 0

    try:
        model, budget, settings = _model(arguments)
    except CannotRun as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CANNOT_RUN

    done: list[RealResult] = []
    try:
        with RealEvalRunner(model, settings, provider=arguments.provider) as runner:
            for item in selected:
                result = runner.run_real(item)
                done.append(result)
                _progress(result, budget)
                if getattr(model, "halt_reason", None):
                    raise RunStopped(model.halt_reason, len(done), len(selected))
    except EvalDatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CANNOT_RUN
    except RunStopped as exc:
        print(
            f"\nSTOPPED after {exc.done} of {exc.total} scenarios: {exc.reason}\n"
            "Nothing was retried. A partial run is not a result and was not written to docs/results; "
            "it will not be resumed without a person.",
            file=sys.stderr,
        )
        _write_partial(done, arguments, exc.reason, settings, budget)
        return STOPPED

    report = score(
        [
            ScenarioResult(scenario=r.scenario.scenario, trace=r.trace, assessment=r.assessment)
            for r in done
            if not r.not_scored
        ],
        REAL_EVAL_VERSION,
    )
    document = results.build(done, report, _meta(arguments, settings, budget, done))
    if arguments.out:
        _write(document, Path(arguments.out), arguments)
    print(results.markdown(document))
    return 0


# --- starting up -------------------------------------------------------------


def _model(arguments: argparse.Namespace):
    base = get_settings()
    if arguments.provider == "stub":
        return StubModel(), None, base

    if arguments.request_cap is None or arguments.max_cost_usd is None:
        raise CannotRun("a real run needs --request-cap and --max-cost-usd, both, every time")
    if arguments.input_usd_per_mtok is None or arguments.output_usd_per_mtok is None:
        raise CannotRun(
            "VoiceDesk ships no prices: give --input-usd-per-mtok and --output-usd-per-mtok (the model's price per million tokens)"
        )
    if not 0 < float(arguments.timeout_seconds) <= 120:
        raise CannotRun("--timeout-seconds must be between 0 and 120")
    try:
        prices = Prices(Decimal(arguments.input_usd_per_mtok), Decimal(arguments.output_usd_per_mtok))
        cap = Decimal(arguments.max_cost_usd)
    except InvalidOperation as exc:
        raise CannotRun("a price or the cost cap is not a number") from exc
    if base.dialogue_max_retries != 0:
        raise CannotRun(
            "adapter retries must be off for a counted run: set VOICEDESK_DIALOGUE_MAX_RETRIES=0 "
            "(an SDK retry is a request the ledger cannot see)"
        )
    if not base.anthropic_api_key:
        raise CannotRun(f"no API key found; the names checked are {', '.join(KEY_NAMES)} (in the environment or a .env)")
    _check_frozen()

    from app.providers.factory import build_model

    budget = RunBudget(
        request_cap=arguments.request_cap, cost_cap_usd=cap, prices=prices, path=Path(arguments.ledger)
    )
    settings = base.model_copy(
        update={
            "llm_input_usd_per_mtok": str(prices.input_per_mtok),
            "llm_output_usd_per_mtok": str(prices.output_per_mtok),
            "dialogue_timeout_seconds": float(arguments.timeout_seconds),
        }
    )
    print(
        f"REAL RUN: {settings.dialogue_model}, timeout {settings.dialogue_timeout_seconds:g}s, retries "
        f"{settings.dialogue_max_retries}. Ledger {arguments.ledger}: {budget.requests_used} of {budget.request_cap} "
        f"requests and {budget.describe_spend()} (cap USD {budget.cost_cap_usd}).",
        flush=True,
    )
    return BudgetedModel(build_model(settings), budget), budget, settings


def _check_frozen() -> None:
    """The scenarios and rules must be the ones recorded in docs/EVAL.md before the run."""
    text = EVAL_DOC.read_text(encoding="utf-8") if EVAL_DOC.is_file() else ""
    found = re.search(r"real-eval-sha256:\s*`?([0-9a-f]{64})`?", text)
    if not found:
        raise CannotRun("docs/EVAL.md records no real-eval-sha256: the scenarios have not been frozen")
    if not freeze.up_to_date():
        raise CannotRun("realeval/scenarios.json is out of date with realeval/scenarios.py")
    if found.group(1) != freeze.digest():
        raise CannotRun("the scenarios or the rules changed since they were frozen (real-eval-sha256 does not match)")


# --- output --------------------------------------------------------------------


def _progress(result: RealResult, budget: RunBudget | None) -> None:
    status = "n/s" if result.not_scored else ("pass" if result.assessment.passed else "FAIL")
    used = f"  [{budget.requests_used} of {budget.request_cap} requests, {budget.describe_spend()}]" if budget else ""
    print(f"  {status:4}  {result.scenario.name}{used}", flush=True)
    if result.not_scored:
        print(f"          NOT SCORED ({result.not_scored}): neither pass nor fail", flush=True)
    for finding in result.assessment.findings:
        if finding.blocking:
            print(f"          {finding.category}: {finding.observed}", flush=True)


def _meta(arguments, settings: Settings, budget: RunBudget | None, done: list[RealResult]) -> dict:
    resolved = next(
        (turn.model_name for r in done for turn in r.trace.turns if turn.model_name), None
    )
    return {
        "run": f"{arguments.provider}-{settings.dialogue_model}",
        "provider": arguments.provider,
        "real_eval_version": REAL_EVAL_VERSION,
        "dataset_sha256": freeze.digest(),
        "model_requested": settings.dialogue_model,
        "model_resolved": resolved,
        "timeout_seconds": settings.dialogue_timeout_seconds,
        "adapter_retries": settings.dialogue_max_retries,
        "thinking": settings.dialogue_thinking,
        "effort": settings.dialogue_effort or None,
        "holding_line_after_seconds": settings.holding_line_after_seconds,
        "temperature": "not set (the API default)",
        "ledger": (
            {
                "path": str(budget.path),
                "request_cap": budget.request_cap,
                "cost_cap_usd": str(budget.cost_cap_usd),
                "requests_used": budget.requests_used,
                "spent_usd_counted": str(budget.spent_usd),
                "requests_of_unknown_cost": budget.unknown_requests,
                "prices_usd_per_mtok": {
                    "input": str(budget.prices.input_per_mtok),
                    "output": str(budget.prices.output_per_mtok),
                },
            }
            if budget
            else None
        ),
        "latency_ms": latency_summary(budget.latencies_ms()) if budget else None,
        "synthetic": True,
    }


def _write(document: dict, out: Path, arguments) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{document['meta']['written_at'][:10]}-voicedesk-real-{arguments.provider}-{document['meta']['model_requested']}"
    (out / f"{stem}.json").write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.md").write_text(results.markdown(document), encoding="utf-8")
    print(f"wrote {out / stem}.json and .md")


def _write_partial(done: list[RealResult], arguments, reason: str, settings=None, budget=None) -> None:
    """Kept outside docs/results, for a person to read: what happened before the stop. Not a result."""
    folder = ROOT / "var" / "real_eval_partial"
    folder.mkdir(parents=True, exist_ok=True)
    if done and settings is not None:
        scored = [
            ScenarioResult(scenario=r.scenario.scenario, trace=r.trace, assessment=r.assessment)
            for r in done
            if not r.not_scored
        ]
        document = results.build(done, score(scored, REAL_EVAL_VERSION), _meta(arguments, settings, budget, done))
        document["meta"]["stopped"] = reason
        (folder / "partial-results.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (folder / "partial-results.md").write_text(results.markdown(document), encoding="utf-8")
    payload = {
        "stopped": reason,
        "completed_scenarios": [r.scenario.name for r in done],
        "last": done[-1].trace.error if done else None,
    }
    (folder / "last-stop.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m realeval", description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="Print the scenarios and run nothing.")
    parser.add_argument("--scenario", action="append", default=None, metavar="NAME")
    parser.add_argument("--provider", choices=("stub", "anthropic"), default="stub")
    parser.add_argument("--request-cap", type=int, default=None)
    parser.add_argument("--max-cost-usd", default=None)
    parser.add_argument("--input-usd-per-mtok", default=None)
    parser.add_argument("--output-usd-per-mtok", default=None)
    parser.add_argument(
        "--timeout-seconds",
        default=EVAL_TIMEOUT_SECONDS,
        type=float,
        help="Model timeout for this evaluation only (default 60). The application keeps its own 15 s.",
    )
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--out", default=None, metavar="DIR", help="Write the JSON and Markdown results here.")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
