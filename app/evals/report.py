"""Printing what the suite measured, and what it did not.

Two audiences. The text report is for a person reading a terminal or a case
study: headline numbers first, then every scenario with the evidence behind
its verdict. The JSON is for a machine, and is stable — categories, keys and
shapes do not move between runs of one suite version.

Both carry the same honesty line, because the number at the top is the one
most likely to be quoted out of context: this is a scripted-model suite, and
it measures VoiceDesk's behaviour rather than any real model's judgement.
"""

import json
from typing import Any

from app.evals.scoring import Percentiles, Ratio, Report, ScenarioResult

HONESTY = (
    "Scripted model, offline providers, isolated database. This measures "
    "VoiceDesk's behaviour under the m9 scenarios; it measures no real "
    "model's conversational quality."
)


def render(report: Report) -> str:
    """The whole report, as text."""
    lines = [
        f"VoiceDesk Evaluation — {report.version}",
        "",
        f"Scenarios: {report.total}",
        f"Passed: {report.passed}",
        f"Failed: {report.failed}",
        f"Task success: {_percent(report.task_success)}",
        "",
        f"Hallucinated availability: {report.hallucinated_availability}",
        "",
        f"Tool correctness: {_percent(report.tool_correctness)}",
        f"  correct: {report.tools.correct}",
        f"  wrong: {report.tools.wrong}",
        f"  missing: {report.tools.missing}",
        f"  duplicate: {report.tools.duplicate}",
        f"  invalid_argument: {report.tools.invalid_argument}",
        "",
        "Escalation:",
        f"  precision: {_fraction(report.escalation_precision)}",
        f"  recall: {_fraction(report.escalation_recall)}",
        "",
        f"Unverified reschedules: {report.unverified_reschedules}",
    ]

    if report.unverified_reschedules:
        lines.append(
            "  observed, not counted against task success: the executor "
            "guards book_appointment and not reschedule, which is frozen "
            "milestone-3/4 behaviour."
        )

    lines += ["", "Turns to booking:", *_percentiles(report.turns_to_booking)]

    lines += [
        "",
        "Cost:",
        f"  measured calls: {report.measured_calls}",
        f"  unpriced calls: {report.unpriced_calls}",
    ]
    if report.total_cost_usd is not None:
        lines.append(f"  total: {report.total_cost_usd} USD")
    if report.unpriced_calls:
        lines.append(
            "  unpriced means no price was configured for a component, not "
            "that it was free."
        )

    if report.authoring_failures:
        lines += [
            "",
            f"Scenario-authoring failures: {report.authoring_failures}",
            "  a fault in the dataset, not in VoiceDesk; the scenario still "
            "counts as failed.",
        ]

    lines += ["", "-" * 62, ""]
    for result in report.results:
        lines.extend(_scenario(result))
        lines.append("")

    lines.append(HONESTY)
    return "\n".join(lines)


def as_json(report: Report) -> str:
    """A stable machine-readable form of the same thing."""
    return json.dumps(to_dict(report), indent=2, sort_keys=False)


def to_dict(report: Report) -> dict[str, Any]:
    return {
        "version": report.version,
        "note": HONESTY,
        "scenarios": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "task_success": _ratio_dict(report.task_success),
        "hallucinated_availability": report.hallucinated_availability,
        "unverified_reschedules": report.unverified_reschedules,
        "authoring_failures": report.authoring_failures,
        "tool_correctness": {
            **_ratio_dict(report.tool_correctness),
            "correct": report.tools.correct,
            "wrong": report.tools.wrong,
            "missing": report.tools.missing,
            "duplicate": report.tools.duplicate,
            "invalid_argument": report.tools.invalid_argument,
        },
        "escalation": {
            "precision": _ratio_dict(report.escalation_precision),
            "recall": _ratio_dict(report.escalation_recall),
        },
        "turns_to_booking": _percentiles_dict(report.turns_to_booking),
        "cost": {
            "measured_calls": report.measured_calls,
            "unpriced_calls": report.unpriced_calls,
            "total_usd": (
                None
                if report.total_cost_usd is None
                else str(report.total_cost_usd)
            ),
        },
        "findings_by_category": report.findings_by_category,
        "results": [_result_dict(result) for result in report.results],
    }


# --- pieces ----------------------------------------------------------------


def _scenario(result: ScenarioResult) -> list[str]:
    verdict = "PASS" if result.passed else "FAIL"
    lines = [
        f"{verdict} {result.name}",
        f"  expected: {result.scenario.expect.task}",
        f"  observed: {_observed(result)}",
        f"  tools: {result.tools_taken}",
        f"  appointments: {_appointments(result)}",
    ]

    for finding in result.assessment.blocking:
        lines.append(f"  finding: {finding.category}")
        lines.append(f"    expected: {finding.expected}")
        lines.append(f"    observed: {finding.observed}")
        if finding.evidence:
            lines.append(f"    evidence: {finding.evidence}")

    for finding in result.assessment.observations:
        lines.append(f"  observation: {finding.category} (non-blocking)")
        lines.append(f"    observed: {finding.observed}")
        if finding.evidence:
            lines.append(f"    evidence: {finding.evidence}")

    return lines


def _observed(result: ScenarioResult) -> str:
    trace = result.trace
    if trace.failed_run:
        return f"the call did not run ({trace.error_category})"
    if trace.escalated:
        return "escalate"
    if trace.booked_appointment_id is not None:
        return "book"
    for record in trace.tool_calls:
        if record.success and record.tool_name in ("reschedule", "cancel"):
            return record.tool_name
        if record.success and record.tool_name == "take_message":
            return "message"
    return "no completed action"


def _appointments(result: ScenarioResult) -> str:
    rows = result.trace.final_appointments
    if not rows:
        return "none"
    return "; ".join(
        f"{row.service_name} for {row.customer_name} at {row.starts_at} "
        f"({row.status})"
        for row in rows
    )


def _percent(ratio: Ratio) -> str:
    if not ratio.defined:
        return "N/A (0 of 0)"
    return f"{ratio.percent:.1f}% ({ratio.numerator}/{ratio.denominator})"


def _fraction(ratio: Ratio) -> str:
    if not ratio.defined:
        return "N/A (nothing to measure)"
    return f"{ratio.numerator}/{ratio.denominator} ({ratio.percent:.1f}%)"


def _percentiles(distribution: Percentiles) -> list[str]:
    if not distribution.count:
        return ["  no booking completed, so there is nothing to measure"]
    lines = [
        f"  count: {distribution.count}",
        f"  p50: {distribution.p50:.0f}",
        f"  p95: {distribution.p95:.0f}",
    ]
    if not distribution.meaningful:
        lines.append("  note: arithmetic, not evidence")
    return lines


def _ratio_dict(ratio: Ratio) -> dict[str, Any]:
    return {
        "numerator": ratio.numerator,
        "denominator": ratio.denominator,
        "percent": None if ratio.percent is None else round(ratio.percent, 1),
    }


def _percentiles_dict(distribution: Percentiles) -> dict[str, Any]:
    return {
        "count": distribution.count,
        "p50": distribution.p50,
        "p95": distribution.p95,
        "meaningful": distribution.meaningful,
        "note": None if distribution.meaningful else "arithmetic, not evidence",
    }


def _result_dict(result: ScenarioResult) -> dict[str, Any]:
    return {
        "scenario": result.name,
        "kind": result.scenario.kind,
        "passed": result.passed,
        "authoring_fault": result.authoring_fault,
        "expected_task": result.scenario.expect.task,
        "observed": _observed(result),
        "tools": [
            {
                "name": record.tool_name,
                "success": record.success,
                "error": record.error,
            }
            for record in result.trace.tool_calls
        ],
        "escalated": result.trace.escalated,
        "booked_turn": result.trace.booked_turn,
        "appointments": [
            {
                "service": row.service_name,
                "customer": row.customer_name,
                "starts_at": row.starts_at,
                "status": row.status,
            }
            for row in result.trace.final_appointments
        ],
        "total_cost_usd": (
            None
            if result.trace.total_cost_usd is None
            else str(result.trace.total_cost_usd)
        ),
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
    }


__all__ = ["HONESTY", "as_json", "render", "to_dict"]
