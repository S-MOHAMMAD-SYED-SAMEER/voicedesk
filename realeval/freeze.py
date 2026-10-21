"""Freezing what a real-model run is judged against.

The scenarios (the caller lines and the outcomes they expect) and the reply rules are written, reviewed and approved
**before** any run, and then fixed: `realeval/scenarios.json` is the reviewable form of the scenarios, and one digest
covers it and `realeval/rules.py`. The digest is recorded in `docs/EVAL.md` (`real-eval-sha256`) and a test fails if
either file changes or if `scenarios.json` is no longer what `realeval/scenarios.py` produces.

    python -m realeval.freeze --write     # regenerate scenarios.json (only while drafting)
    python -m realeval.freeze --check     # fail if it is out of date
    python -m realeval.freeze             # print the digest
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from realeval import REAL_EVAL_VERSION
from realeval.scenarios import SCENARIOS, RealScenario

ROOT = Path(__file__).resolve().parent
SCENARIOS_JSON = ROOT / "scenarios.json"
RULES = ROOT / "rules.py"


def _spec(item: RealScenario) -> dict[str, Any]:
    scenario = item.scenario
    expect = scenario.expect
    return {
        "name": scenario.name,
        "description": scenario.description,
        "now": item.now.isoformat(),
        "world": {
            "services": [
                {"name": s.name, "duration_minutes": s.duration_minutes, "staff": s.staff_id}
                for s in scenario.world.services
            ],
            "hours": {
                "weekdays": sorted({h.weekday for h in scenario.world.hours}),
                "opens": scenario.world.hours[0].opens_at.isoformat(timespec="minutes"),
                "closes": scenario.world.hours[0].closes_at.isoformat(timespec="minutes"),
            },
            "already_booked": [
                {
                    "ref": seed.ref,
                    "service": seed.service_name,
                    "customer": seed.customer_name,
                    "phone": seed.phone,
                    "starts_at": seed.starts_at,
                    "cancelled": seed.cancelled,
                }
                for seed in scenario.world.appointments
            ],
        },
        "caller_lines": [beat.caller for beat in scenario.script],
        "expect": {
            "task": expect.task,
            "must_call": [
                {"tool": tool.name, "arguments": dict(tool.arguments), "succeeds": tool.succeeds}
                for tool in expect.expected_tools
            ],
            "must_not_call": sorted(expect.forbidden_tools),
            "may_fail": sorted(expect.allow_tool_failures),
            "read_only_tools_tolerated": sorted(expect.tolerated_tools),
            "must_escalate": expect.should_escalate,
            "appointment_book_at_the_end": [
                {
                    "service": spec.service_name,
                    "customer": spec.customer_name,
                    "starts_at": spec.starts_at,
                    "status": spec.status,
                }
                for spec in expect.final_appointments
            ],
            "staff_requests_recorded": (
                None if item.staff_requests is None else sorted(item.staff_requests)
            ),
            "escalation_tolerated": item.escalation_tolerated,
        },
    }


def scenarios_text() -> str:
    document = {"version": REAL_EVAL_VERSION, "scenarios": [_spec(item) for item in SCENARIOS]}
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _normalised(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def digest() -> str:
    """sha256 over the scenarios file and the rules file (line endings normalised, so a checkout on any OS agrees)."""
    sha = hashlib.sha256()
    for path in (SCENARIOS_JSON, RULES):
        sha.update(path.name.encode() + b"\0" + _normalised(path) + b"\0")
    return sha.hexdigest()


def up_to_date() -> bool:
    return SCENARIOS_JSON.read_bytes().replace(b"\r\n", b"\n").decode("utf-8") == scenarios_text()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--write" in args:
        SCENARIOS_JSON.write_text(scenarios_text(), encoding="utf-8", newline="\n")
        print(f"wrote {SCENARIOS_JSON}")
    elif "--check" in args:
        if not up_to_date():
            print("realeval/scenarios.json is out of date with realeval/scenarios.py", file=sys.stderr)
            return 1
    print(digest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
