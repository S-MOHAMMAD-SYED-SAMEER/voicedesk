"""The turn manager: text in, text out, with tools in between.

One `Conversation` is one call. `send()` takes what the caller said and
returns what the receptionist says back, having run whatever tools the model
asked for along the way. There is no audio here and none is coming — the
specification is explicit that if testing a booking flow needs a phone call,
the layering is wrong.

Two invariants hold whatever the model does:

* **The turn always ends.** The loop is bounded. Reaching the bound is a
  failure with a fixed reply, never an answer.
* **A failure is never dressed up as a success.** If the model cannot be
  reached, or the loop runs out, the caller is told something safe and
  `DialogueResult.failed` is true. Nothing in this module can report a
  booking, a cancellation or a free slot that a tool did not actually
  produce.

Conversation history lives here, in memory. It contains the provider's tool
use and tool result blocks, which `turns` deliberately does not store — that
table is the readable transcript, not a replay log. Milestone 5 will hold a
`Conversation` for the length of a call.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.dialogue.definitions import tool_definitions
from app.dialogue.errors import ToolLoopExhausted
from app.dialogue.executor import ToolCallRecord, ToolExecutor
from app.dialogue.prompt import SYSTEM_PROMPT_VERSION, build_system_prompt
from app.models import Call, ToolCall, Turn, TurnRole
from app.providers.llm import LanguageModel, Message, ModelError
from app.tools import ToolContext

# Said when the system cannot answer. Fixed strings, because a model that has
# just failed is not the thing to ask for an apology.
MODEL_FAILURE_REPLY = (
    "I'm sorry, I'm having trouble with our system just now. "
    "Let me pass you to a colleague."
)
LOOP_EXHAUSTED_REPLY = (
    "I'm sorry, I'm not getting anywhere with that. "
    "Let me pass you to a colleague who can help."
)


@dataclass(frozen=True)
class DialogueResult:
    """What one caller turn produced."""

    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    escalated: bool = False
    booked_appointment_id: str | None = None
    failed: bool = False
    llm_latency_ms: int = 0
    prompt_version: str = SYSTEM_PROMPT_VERSION


class Conversation:
    """One call's dialogue: history, the model, the tools, the transcript."""

    def __init__(
        self,
        session: Session,
        call: Call,
        model: LanguageModel,
        settings: Settings | None = None,
    ) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._model = model
        self._call_id = call.id
        self._history: list[Message] = []
        self._tools = tool_definitions()
        # Read once, at the start of the call: the menu is what the business
        # offers, not something that should change mid-conversation.
        self._system = build_system_prompt(session)
        self._executor = ToolExecutor(
            ToolContext(
                session=session, settings=self._settings, call_id=self._call_id
            )
        )

    @property
    def system_prompt(self) -> str:
        return self._system

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    def send(self, caller_text: str) -> DialogueResult:
        """One caller turn, through the model and its tools, to a reply."""
        spoken_at = datetime.now(UTC)
        self._history.append(
            Message(role="user", content=[{"type": "text", "text": caller_text}])
        )

        records: list[ToolCallRecord] = []
        latencies: list[int] = []
        failed = False

        try:
            reply = self._run(records, latencies)
        except ModelError:
            reply, failed = MODEL_FAILURE_REPLY, True
        except ToolLoopExhausted:
            reply, failed = LOOP_EXHAUSTED_REPLY, True

        latency_ms = sum(latencies)
        self._persist(caller_text, spoken_at, reply, latency_ms, records)
        return DialogueResult(
            text=reply,
            tool_calls=records,
            escalated=_escalated(records),
            booked_appointment_id=_booked(records),
            failed=failed,
            llm_latency_ms=latency_ms,
        )

    def _run(self, records: list[ToolCallRecord], latencies: list[int]) -> str:
        """The model/tool loop, to a final reply or an exception.

        Both exits are failures the caller turn can survive: a `ModelError`
        from the provider, or `ToolLoopExhausted` when the model keeps asking
        for tools. `records` and `latencies` are filled as it goes, so a turn
        that ends either way still has a transcript to write.
        """
        for _ in range(self._settings.max_tool_iterations):
            response = self._model.respond(
                system=self._system,
                messages=self._history,
                tools=self._tools,
            )
            latencies.append(response.latency_ms or 0)
            self._history.append(
                Message(role="assistant", content=response.raw_content)
            )

            if not response.tool_uses:
                return response.text

            blocks = []
            for tool_use in response.tool_uses:
                block, record = self._executor.execute(tool_use)
                blocks.append(block)
                records.append(record)
            # Every result from one response goes back in a single message:
            # splitting them teaches the model to stop asking for tools in
            # parallel.
            self._history.append(Message(role="user", content=blocks))

        raise ToolLoopExhausted(
            f"The model asked for tools {self._settings.max_tool_iterations} "
            "times without finishing the turn."
        )

    def _persist(
        self,
        caller_text: str,
        spoken_at: datetime,
        reply: str,
        latency_ms: int,
        records: list[ToolCallRecord],
    ) -> None:
        """Write the transcript: one caller turn, one agent turn, its tools.

        Both timestamps are set here rather than by the database, because
        PostgreSQL's `now()` is the transaction's time — two rows written
        together would share it, and `Call.turns` orders by `created_at`.

        A failed turn is persisted like any other. A call that went wrong is
        exactly the one somebody will want to read afterwards.
        """
        caller_turn = Turn(
            call_id=self._call_id,
            role=TurnRole.CALLER,
            text=caller_text,
            created_at=spoken_at,
        )
        agent_turn = Turn(
            call_id=self._call_id,
            role=TurnRole.AGENT,
            text=reply,
            llm_latency_ms=latency_ms or None,
            created_at=datetime.now(UTC),
        )
        agent_turn.tool_calls = [
            ToolCall(
                tool_name=record.tool_name,
                arguments=record.arguments,
                # A failed call has no result to record, only an error. The
                # recovery data in `record.data` has already gone back to the
                # model in the tool_result block; it is not an outcome.
                result=record.data if record.success else None,
                success=record.success,
                error=record.error,
                latency_ms=record.latency_ms,
            )
            for record in records
        ]

        self._session.add_all([caller_turn, agent_turn])
        self._session.commit()


def _escalated(records: list[ToolCallRecord]) -> bool:
    return any(
        record.tool_name == "transfer_to_human" and record.success
        for record in records
    )


def _booked(records: list[ToolCallRecord]) -> str | None:
    for record in records:
        if record.tool_name == "book_appointment" and record.success:
            appointment_id = record.data.get("appointment_id")
            if appointment_id is not None:
                return str(appointment_id)
    return None
