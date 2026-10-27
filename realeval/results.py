"""What a real-model run wrote down: every scenario, every reply, every tool call, every finding.

Two files per run: JSON (the whole record, for tests and for the case study to read figures from) and a Markdown
summary. Nothing here is a claim: the Markdown opens with what the run was and was not.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.evals.scoring import Report
from realeval.runner import RealResult
from realeval.scenarios import SCENARIOS

_CALLER_TURNS = sum(len(item.scenario.script) for item in SCENARIOS)

LIMITS = (
    f"Synthetic salon, {len(SCENARIOS)} scenarios ({_CALLER_TURNS} caller turns), scripted caller lines that cannot adapt, one model, one run, default sampling. "
    f"{sum(item.safety_only for item in SCENARIOS)} of them are safety-only: a receptionist that does nothing passes them, so they show that no harm was done, not that the job was done. "
    "Typed text: no speech, no telephony, no real caller; fictional salon data. Expectations drafted by Claude; "
    "reviewed by ChatGPT (on the 16-scenario draft) and Claude; changes made in response; final decisions by Sameer; "
    "frozen before the run. Latency is reported apart and is not graded: the timeout here is 60 s for the evaluation "
    "only, and real phone use needs a much faster reply, which this test does not measure. Not an accuracy figure and "
    "not evidence of how the receptionist behaves on a real call."
)


def _tool(record: Any) -> dict[str, Any]:
    return {
        "tool": record.tool_name,
        "arguments": record.arguments,
        "success": record.success,
        "error": record.error,
        "latency_ms": record.latency_ms,
    }


def _scenario(result: RealResult) -> dict[str, Any]:
    trace = result.trace
    return {
        "name": result.scenario.name,
        "description": result.scenario.scenario.description,
        "now": result.scenario.now.isoformat(),
        "passed": None if result.not_scored else result.assessment.passed,
        "not_scored": result.not_scored,
        "safety_only": result.scenario.safety_only,
        "findings": [
            {
                "category": finding.category,
                "blocking": finding.blocking,
                "expected": finding.expected,
                "observed": finding.observed,
                "evidence": finding.evidence,
            }
            for finding in result.assessment.findings
        ],
        "turns": [
            {
                "caller": turn.caller_text,
                "reply": turn.reply,
                "tool_calls": [_tool(record) for record in turn.tool_calls],
                "failed": turn.failed,
                "escalated": turn.escalated,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "llm_latency_ms": turn.llm_latency_ms,
            }
            for turn in trace.turns
        ],
        "model_requests": trace.model_calls,
        "final_appointments": [
            {
                "service": spec.service_name,
                "customer": spec.customer_name,
                "starts_at": spec.starts_at,
                "status": spec.status,
            }
            for spec in trace.final_appointments
        ],
        "staff_requests": result.staff_requests,
        "observations": [
            {"category": f.category, "observed": f.observed, "evidence": f.evidence}
            for f in result.observations
        ],
        "cost_usd": str(trace.total_cost_usd) if trace.total_cost_usd is not None else None,
        "error": trace.error,
    }


def build(results: list[RealResult], report: Report, meta: dict[str, Any]) -> dict[str, Any]:
    scored = [r for r in results if not r.not_scored]
    passed = sum(1 for r in scored if r.assessment.passed)
    unflagged = [
        finding.category
        for r in scored
        for finding in r.assessment.findings
        if finding.blocking
    ]
    return {
        "meta": {**meta, "written_at": datetime.now(UTC).isoformat(), "limits": LIMITS},
        "summary": {
            "scenarios": len(results),
            "passed": passed,
            "failed": len(scored) - passed,
            "not_scored": len(results) - len(scored),
            "not_scored_names": [r.scenario.name for r in results if r.not_scored],
            "safety_only_passed": sum(1 for r in scored if r.scenario.safety_only and r.assessment.passed),
            "safety_only_total": sum(1 for r in results if r.scenario.safety_only),
            "task_passed": sum(1 for r in scored if not r.scenario.safety_only and r.assessment.passed),
            "task_total": sum(1 for r in results if not r.scenario.safety_only),
            "blocking_findings": len(unflagged),
            "findings_by_category": {c: unflagged.count(c) for c in sorted(set(unflagged))},
            "observations": sum(len(r.observations) for r in results),
            "hallucinated_availability": report.hallucinated_availability,
            "model_requests": sum(r.trace.model_calls for r in results),
            "cost_usd": str(
                sum((r.trace.total_cost_usd or Decimal(0) for r in results), Decimal(0))
            ),
        },
        "scenarios": [_scenario(result) for result in results],
    }


def _ledger_cost(ledger: dict | None) -> str:
    """The ledger's figure counts a request that got no answer at its maximum, never at zero."""
    if not ledger:
        return ""
    unknown = ledger.get("requests_of_unknown_cost", 0)
    text = f"; the ledger counts USD {ledger['spent_usd_counted']} of USD {ledger['cost_cap_usd']}"
    if unknown:
        text += f", including {unknown} request(s) of unknown cost counted at their maximum"
    return text


def _latency_line(latency: dict | None, timeout: float | None) -> str:
    """Reported, never graded. The 60 s limit of an evaluation says nothing about real phone use."""
    if not latency or not latency.get("requests"):
        return "Latency: not measured."
    return (
        f"Latency per model request (reported, not graded): p50 {latency['p50_ms']} ms, p95 {latency['p95_ms']} ms over "
        f"{latency['requests']} request(s). The timeout here was {timeout:g} s for the evaluation only; real phone use "
        "needs a much faster reply, which this test does not measure."
    )


def markdown(document: dict[str, Any]) -> str:
    meta, summary = document["meta"], document["summary"]
    lines = [
        f"# VoiceDesk real-model run: {meta.get('run', '')}",
        "",
        f"**Read this first.** {meta['limits']}",
        "",
        f"Model: `{meta.get('model_requested')}` (resolved by the API: `{meta.get('model_resolved')}`). "
        f"Scenario pack `{meta.get('real_eval_version')}` (`real-eval-sha256` {meta.get('dataset_sha256')}). "
        f"Provider: {meta.get('provider')}.",
        "",
        f"**{summary['passed']} of {summary['scenarios']} scenarios passed**: "
        f"{summary['task_passed']} of {summary['task_total']} task scenarios and "
        f"{summary['safety_only_passed']} of {summary['safety_only_total']} safety-only scenarios. "
        f"{summary['blocking_findings']} blocking finding(s) {summary['findings_by_category']}; "
        f"{summary['model_requests']} model request(s); USD {summary['cost_usd']} from the requests that were answered"
        + _ledger_cost(meta.get("ledger"))
        + ". "
        + (
            f"**{summary['not_scored']} scenario(s) not scored (timeout), neither pass nor fail: "
            f"{', '.join(summary['not_scored_names'])}.**"
            if summary["not_scored"]
            else "No scenario was left unscored."
        ),
        "",
        _latency_line(meta.get("latency_ms"), meta.get("timeout_seconds")),
        "",
        "| Scenario | Kind | Result | Findings |",
        "|---|---|---|---|",
    ]
    for scenario in document["scenarios"]:
        found = "; ".join(f"{f['category']}: {f['observed']}" for f in scenario["findings"] if f["blocking"]) or "none"
        kind = "safety-only" if scenario["safety_only"] else "task"
        verdict = "not scored (timeout)" if scenario.get("not_scored") else ("pass" if scenario["passed"] else "**FAIL**")
        lines.append(f"| {scenario['name']} | {kind} | {verdict} | {found} |")
    return "\n".join(lines) + "\n"
