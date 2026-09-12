"""Running one scenario against the real system, and watching what happens.

    world → ScriptedModel → Conversation → ToolExecutor → tools
                                                            ↓
                                              CalendarService → PostgreSQL

Only the model is scripted. The dialogue layer, the executor and its
offered-slot guard, all six tools, the calendar and the database's exclusion
constraint are the ones the application runs. Nothing is stubbed to make a
scenario pass, and a scenario that fails because the real system refused it is
the suite working.

Two kinds of failure are told apart deliberately. A model that runs out of
script is a **dataset** fault (`SCRIPT_EXHAUSTED`); anything else that stops a
call before it can be judged is a **system** fault (`PROVIDER_FAILURE`). They
are reported under different headings because confusing them would let a
half-written scenario masquerade as a receptionist misbehaving.

Cost comes from milestone 8 unchanged: the runner calls the same public
recorder a transport calls. A text evaluation buys no speech, so it reports no
speech usage — an empty provider name writes no row, rather than a zero one.
"""

import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal

import anyio
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.cost import record_turn_cost
from app.dialogue import Conversation
from app.evals import database as evaldb
from app.evals import realtime as rt
from app.evals import world as worlds
from app.evals.model import ScriptedModel, ScriptExhausted
from app.evals.scenario import (
    EVAL_TIMEZONE,
    AppointmentSpec,
    Scenario,
    ScenarioError,
    timezone,
)
from app.evals.trace import PROVIDER_FAILURE, SCRIPT_EXHAUSTED, CallTrace, TurnTrace
from app.models import Appointment, Call, CallCost, Service
from app.providers.offline_streaming import OfflineStreamingSpeechToText
from app.realtime import RealtimeSession
from app.realtime.sink import NullSink

logger = logging.getLogger(__name__)

# Recorded as the model provider on every cost row the suite writes. No
# vendor answered, so no vendor is named: a row naming one would be a claim
# about spending that never happened.
SCRIPTED_PROVIDER = "scripted"


@dataclass(frozen=True)
class NoSpeechUsage:
    """What a text evaluation bought from a speech provider: nothing.

    Empty provider names, so milestone 8's recorder writes no speech rows at
    all. Zeroed usage would claim a provider had been asked and had charged
    nothing, which is a different and false statement.
    """

    stt_provider: str = ""
    stt_audio_ms: int | None = None
    tts_provider: str = ""
    tts_characters: int | None = None


class EvalRunner:
    """One migrated evaluation database, and every scenario run against it."""

    def __init__(self, settings: Settings | None = None) -> None:
        base = settings or get_settings()
        self._url = evaldb.resolve_eval_database_url(base)
        # Cost tracking on, so milestone 8 writes what it writes. Timezone
        # fixed, so the dataset's dates mean the same thing on every machine.
        self._settings = base.model_copy(
            update={
                "database_url": self._url,
                "cost_tracking_enabled": True,
                "business_timezone": EVAL_TIMEZONE,
            }
        )
        self._engine: Engine | None = None

    @property
    def url(self) -> str:
        return self._url

    @property
    def settings(self) -> Settings:
        return self._settings

    def __enter__(self) -> "EvalRunner":
        evaldb.ensure_database(self._url)
        evaldb.migrate(self._url)
        self._engine = create_engine(self._url)
        return self

    def __exit__(self, *exception: object) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def run(self, scenario: Scenario) -> CallTrace:
        """One scenario, from an empty database to a finished trace."""
        if self._engine is None:
            raise RuntimeError("Use EvalRunner as a context manager.")

        evaldb.truncate(self._engine)
        trace = CallTrace(scenario=scenario.name, kind=scenario.kind)

        with Session(self._engine) as session:
            try:
                if scenario.kind == "realtime":
                    self._realtime(session, scenario, trace)
                else:
                    self._text(session, scenario, trace)
            except ScriptExhausted as exc:
                trace.error, trace.error_category = str(exc), SCRIPT_EXHAUSTED
            except ScenarioError as exc:
                trace.error, trace.error_category = str(exc), SCRIPT_EXHAUSTED
            except Exception as exc:  # noqa: BLE001 - one scenario must not end a run
                logger.exception("Scenario %s could not be run.", scenario.name)
                trace.error = f"{type(exc).__name__}: {exc}"
                trace.error_category = PROVIDER_FAILURE

            self._finalise(session, trace)

        return trace

    # --- the text path -----------------------------------------------------

    def _text(self, session: Session, scenario: Scenario, trace: CallTrace) -> None:
        model, call, appointments = self._prepare(session, scenario, trace)
        conversation = Conversation(session, call, model, self._settings)

        for index, beat in enumerate(scenario.script):
            result = conversation.send(beat.caller)
            trace.turns.append(_turn_trace(index, beat.caller, result))
            self._record_cost(session, result, NoSpeechUsage())

        trace.model_calls = model.call_count

    # --- the realtime path -------------------------------------------------

    def _realtime(self, session: Session, scenario: Scenario, trace: CallTrace) -> None:
        model, call, appointments = self._prepare(session, scenario, trace)
        conversation = Conversation(session, call, model, self._settings)
        tts = rt.PacedTextToSpeech()
        sink = NullSink()

        voice = RealtimeSession(
            conversation=conversation,
            stt=OfflineStreamingSpeechToText(),
            tts=tts,
            sink=sink,
            settings=self._settings.model_copy(update={"realtime_enabled": True}),
            sample_rate=rt.SAMPLE_RATE,
        )

        anyio.run(self._drive, voice, scenario, trace)

        trace.model_calls = model.call_count
        trace.generation = voice.generation
        trace.sink_chunks = len(sink.chunks)
        trace.sink_clears = sink.clears

        for index, turn in enumerate(voice.turns):
            trace.turns.append(_realtime_turn_trace(index, turn))
            if turn.dialogue is not None:
                self._record_cost(session, turn.dialogue, turn)

    async def _drive(
        self, voice: RealtimeSession, scenario: Scenario, trace: CallTrace
    ) -> None:
        """Feed the session frames, and interrupt it if the scenario says to."""
        speech = rt.speech_frame()
        quiet = rt.silent_frame()

        async with anyio.create_task_group() as group:
            voice.attach(group)
            await rt.feed(voice, scenario.speech_frames, speech)
            await rt.feed(voice, scenario.silence_frames, quiet)

            if scenario.interrupt:
                # The reply is held at its first chunk; talking over it now is
                # an interruption rather than a comment on a finished sentence.
                await anyio.sleep(rt.SETTLE_SECONDS)
                await rt.feed(voice, scenario.speech_frames, speech)
            else:
                await voice.finish()

            group.cancel_scope.cancel()

    # --- shared ------------------------------------------------------------

    def _prepare(
        self, session: Session, scenario: Scenario, trace: CallTrace
    ) -> tuple[ScriptedModel, Call, dict[str, uuid.UUID]]:
        appointments = worlds.build(session, scenario.world, self._settings)
        call = worlds.start_call(session, scenario.world)
        trace.call_id = call.id
        trace.appointment_refs = {
            ref: str(identifier) for ref, identifier in appointments.items()
        }
        responses = worlds.scripted_responses(scenario, appointments)
        return ScriptedModel(responses, scenario=scenario.name), call, appointments

    def _record_cost(self, session: Session, result: object, speech: object) -> None:
        """Milestone 8's recorder, called exactly as a transport calls it."""
        record_turn_cost(
            session,
            result,
            speech,
            self._settings,
            llm_provider=SCRIPTED_PROVIDER,
        )

    def _finalise(self, session: Session, trace: CallTrace) -> None:
        """Read back what the database actually holds now.

        Expunged, then rolled back. A scenario that ended mid-flush leaves
        objects carrying changes the database refused, and a plain rollback
        would warn about discarding them; dropping them first says the same
        thing more quietly. Nothing committed is lost — the tools and the
        dialogue layer commit their own work.
        """
        session.expunge_all()
        session.rollback()
        trace.final_appointments = _appointments(session)
        if trace.call_id is None:
            return

        call = session.get(Call, trace.call_id)
        if call is not None:
            trace.total_cost_usd = call.total_cost_usd
        trace.component_costs = _component_costs(session, trace.call_id)


# --- reading the world back ------------------------------------------------


def _appointments(session: Session) -> list[AppointmentSpec]:
    """Every appointment, as the ground truth spells them."""
    rows = session.execute(
        select(Appointment, Service.name)
        .join(Service, Appointment.service_id == Service.id)
        .order_by(Appointment.starts_at, Appointment.customer_name)
    ).all()
    zone = timezone()
    return [
        AppointmentSpec(
            service_name=name,
            customer_name=appointment.customer_name,
            starts_at=appointment.starts_at.astimezone(zone).isoformat(),
            status=str(appointment.status),
        )
        for appointment, name in rows
    ]


def _component_costs(
    session: Session, call_id: uuid.UUID
) -> dict[str, Decimal | None]:
    rows = session.execute(
        select(CallCost).where(CallCost.call_id == call_id)
    ).scalars()
    return {str(row.component): row.cost_usd for row in rows}


def _turn_trace(index: int, caller: str, result) -> TurnTrace:
    return TurnTrace(
        index=index,
        caller_text=caller,
        reply=result.text,
        tool_calls=list(result.tool_calls),
        escalated=result.escalated,
        booked_appointment_id=result.booked_appointment_id,
        failed=result.failed,
        model_name=result.model_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        llm_latency_ms=result.llm_latency_ms,
    )


def _realtime_turn_trace(index: int, turn) -> TurnTrace:
    result = turn.dialogue
    trace = TurnTrace(
        index=index,
        caller_text=turn.transcript,
        reply=turn.reply,
        generation=turn.generation,
        interrupted=turn.interrupted,
        failed=turn.failed,
    )
    if result is not None:
        trace.tool_calls = list(result.tool_calls)
        trace.escalated = result.escalated
        trace.booked_appointment_id = result.booked_appointment_id
        trace.model_name = result.model_name
        trace.input_tokens = result.input_tokens
        trace.output_tokens = result.output_tokens
        trace.llm_latency_ms = result.llm_latency_ms
    return trace


__all__ = ["SCRIPTED_PROVIDER", "EvalRunner", "NoSpeechUsage"]
