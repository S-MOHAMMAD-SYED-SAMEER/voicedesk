"""One real-model scenario at a time, over the same machinery as the scripted suite.

Everything below the model is `app.evals`: the migrated evaluation database, the world built through the real
`CalendarService`, the real dialogue layer, tools and calendar, the trace and the checks. What changes is what is behind
`LanguageModel` (a budgeted real model, or an offline stub), the clock (each scenario says what time it is), and two
reads at the end: the staff requests recorded for the call, and the reply rules of `realeval.rules`.
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.clock import fixed_clock
from app.config import Settings
from app.dialogue import Conversation
from app.evals import world as worlds
from app.evals.checks import (
    UNDECLARED_CLAIM,
    UNNECESSARY_ESCALATION,
    Assessment,
    Finding,
    assess,
)
from app.evals.runner import EvalRunner, NoSpeechUsage, _turn_trace
from app.evals.scenario import Scenario
from app.evals.trace import CallTrace
from app.models import Call, StaffRequest
from app.providers.llm import LanguageModel
from realeval.rules import reply_findings, staff_request_findings
from realeval.scenarios import RealScenario

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"

#: A reply that names a clock time the calendar did not offer ("10:00 is taken") is **not** a failure for a real model:
#: it repeats the time it is declining. The scripted suite counts it because every scripted reply is written to avoid
#: it. Here it is an observation, listed with the reply so a person can read it; the structural check (no booking for
#: a slot the calendar never offered) stays blocking.
OBSERVATION_CATEGORIES = frozenset({UNDECLARED_CLAIM})


@dataclass
class RealResult:
    scenario: RealScenario
    trace: CallTrace
    assessment: Assessment
    staff_requests: list[dict[str, str | None]] = field(default_factory=list)
    #: Things noticed and written down but not counted against the scenario (see `OBSERVATION_CATEGORIES`).
    observations: list[Finding] = field(default_factory=list)
    halted: str | None = None


class RealEvalRunner(EvalRunner):
    """`EvalRunner` with a model and a clock supplied from outside."""

    def __init__(self, model: LanguageModel, settings: Settings | None = None, *, provider: str = PROVIDER) -> None:
        super().__init__(settings)
        self._model = model
        self._provider = provider
        self._now = None
        self._staff: list[dict[str, str | None]] = []

    # --- one scenario, scored ---------------------------------------------

    def run_real(self, item: RealScenario) -> RealResult:
        self._now = item.now
        self._staff = []
        start = getattr(self._model, "call_count", 0)
        trace = self.run(item.scenario)
        trace.model_calls = getattr(self._model, "call_count", 0) - start
        assessment = assess(item.scenario, trace)
        observed = set(OBSERVATION_CATEGORIES)
        if item.escalation_tolerated:
            observed.add(UNNECESSARY_ESCALATION)
        observations = [f for f in assessment.findings if f.category in observed]
        assessment.findings = [f for f in assessment.findings if f.category not in observed]
        if not trace.failed_run:
            assessment.findings.extend(
                reply_findings(trace, disclosure_forbidden=item.disclosure_forbidden)
            )
            assessment.findings.extend(
                staff_request_findings(
                    item.staff_requests,
                    [row["kind"] for row in self._staff],
                    item.optional_staff_requests,
                )
            )
        return RealResult(
            scenario=item,
            trace=trace,
            assessment=assessment,
            staff_requests=list(self._staff),
            observations=observations,
            halted=getattr(self._model, "halt_reason", None),
        )

    # --- hooks the scripted runner already has ------------------------------

    def _prepare(self, session: Session, scenario: Scenario, trace: CallTrace):
        appointments = worlds.build(session, scenario.world, self._settings)
        call = worlds.start_call(session, scenario.world)
        trace.call_id = call.id
        trace.appointment_refs = {ref: str(identifier) for ref, identifier in appointments.items()}
        return self._model, call, appointments

    def _text(self, session: Session, scenario: Scenario, trace: CallTrace) -> None:
        model, call, _ = self._prepare(session, scenario, trace)
        conversation = Conversation(
            session, call, model, self._settings, clock=fixed_clock(self._now)
        )
        for index, beat in enumerate(scenario.script):
            result = conversation.send(beat.caller)
            trace.turns.append(_turn_trace(index, beat.caller, result))
            self._record_cost(session, result, NoSpeechUsage())
            if getattr(model, "halt_reason", None):
                # The first stop-rule error ends the run; nothing more is sent.
                break

    def _record_cost(self, session: Session, result: object, speech: object) -> None:
        from app.cost import record_turn_cost

        record_turn_cost(session, result, speech, self._settings, llm_provider=self._provider)

    def _finalise(self, session: Session, trace: CallTrace) -> None:
        if trace.call_id is not None:
            call = session.get(Call, trace.call_id)
            rows = (
                session.execute(
                    select(StaffRequest)
                    .where(StaffRequest.call_id == trace.call_id)
                    .order_by(StaffRequest.created_at)
                )
                .scalars()
                .all()
            )
            self._staff = [
                {
                    "kind": str(row.kind),
                    "caller_name": row.caller_name,
                    "phone": row.phone,
                    "body": row.body,
                }
                for row in rows
            ]
            del call
        super()._finalise(session, trace)


__all__ = ["PROVIDER", "RealEvalRunner", "RealResult"]
