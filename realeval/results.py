"""What a real-model run wrote down: every scenario, every reply, every tool call, every finding.

Two files per run: JSON (the whole record, for tests and for the case study to read figures from) and a Markdown
summary. Nothing here is a claim: the Markdown opens with what the run was and was not.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.evals.scoring import Report
from realeval.runner import RealResult

LIMITS = (
    "Synthetic salon, 16 scenarios (27 caller turns), scripted caller lines that cannot adapt, one model, one run, default sampling. "
    "Typed text: no speech, no telephony, no real caller. Expectations drafted by Claude, reviewed by ChatGPT and "
    "Claude, approved by Sameer, and frozen before the run. Not an accuracy figure and not evidence of how the "
    "receptionist behaves on a real call."
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
        "passed": result.assessment.passed,
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
    passed = sum(1 for r in results if r.assessment.passed)
    unflagged = [
        finding.category
        for r in results
        for finding in r.assessment.findings
        if finding.blocking
    ]
    return {
        "meta": {**meta, "written_at": datetime.now(UTC).isoformat(), "limits": LIMITS},
        "summary": {
            "scenarios": len(results),
            "passed": passed,
            "failed": len(results) - passed,
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
        f"**{summary['passed']} of {summary['scenarios']} scenarios passed**; {summary['blocking_findings']} blocking "
        f"finding(s) {summary['findings_by_category']}; {summary['model_requests']} model request(s); "
        f"USD {summary['cost_usd']}.",
        "",
        "| Scenario | Result | Findings |",
        "|---|---|---|",
    ]
    for scenario in document["scenarios"]:
        found = "; ".join(f"{f['category']}: {f['observed']}" for f in scenario["findings"] if f["blocking"]) or "none"
        lines.append(f"| {scenario['name']} | {'pass' if scenario['passed'] else '**FAIL**'} | {found} |")
    return "\n".join(lines) + "\n"
