"""A language model that says exactly what the scenario told it to.

    ScriptedModel → Conversation → ToolExecutor → tools → CalendarService → PostgreSQL

Only the top of that chain is fake. Everything below it is the real thing, and
that is the point: the suite measures what VoiceDesk does with a given set of
model outputs, against a real calendar and a real database with its real
constraints.

**What this does not measure.** A scripted model has no opinion, makes no
mistakes and cannot be surprised. Nothing this model does says anything about
how Claude would behave on a real call, and no result produced with it should
ever be described as a measurement of a real model's conversational quality.

Running out of script is a **dataset** fault, not a VoiceDesk one. It raises
`ScriptExhausted`, which the runner catches and reports under its own heading
so that a half-written scenario can never be mistaken for the receptionist
misbehaving.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from app.providers.llm import Message, ModelResponse, ToolDefinition


class ScriptExhausted(RuntimeError):
    """The scenario ran out of scripted responses mid-call."""


class ScriptedModel:
    """A `LanguageModel` whose answers were written in advance.

    Deterministic by construction: the same scenario produces the same
    responses in the same order on every run, on any machine, with no network
    and no credential.
    """

    def __init__(
        self,
        responses: Iterable[ModelResponse] = (),
        *,
        scenario: str = "",
    ) -> None:
        self._responses = list(responses)
        self._scenario = scenario
        self._index = 0
        # Kept so a scenario can assert that the model was not asked at all —
        # which is what silence must produce.
        self.requests: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def remaining(self) -> int:
        return len(self._responses) - self._index

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def respond(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """The next scripted response, and a record of what was asked."""
        self.requests.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": list(tools),
            }
        )
        if self._index >= len(self._responses):
            raise ScriptExhausted(
                f"{self._scenario or 'This scenario'} ran out of scripted "
                f"responses after {self._index} of them. The script must "
                "cover every request the dialogue layer makes, including each "
                "trip round the tool loop."
            )
        response = self._responses[self._index]
        self._index += 1
        return response




# --- writing a script ------------------------------------------------------

# Fixed, so two runs of one scenario record the same usage and the same cost.
# They are not a measurement of anything: no model was asked, so these are the
# scenario's own numbers and the cost rows they produce say `scripted`.
INPUT_TOKENS = 400
OUTPUT_TOKENS = 40

MODEL_NAME = "scripted-model"


def say(text: str, *, latency_ms: int = 10) -> ModelResponse:
    """A plain reply, ending the turn."""
    return ModelResponse(
        text=text,
        stop_reason="end_turn",
        raw_content=[{"type": "text", "text": text}],
        model_name=MODEL_NAME,
        latency_ms=latency_ms,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )


def use_tools(
    *calls: tuple[str, dict[str, Any]], text: str = "", latency_ms: int = 10
) -> ModelResponse:
    """A reply that asks for one or more tools.

    Identifiers are derived from the tool name and its position rather than
    generated, because a scenario that produced different bytes on every run
    could not be compared with itself.
    """
    from app.providers.llm import ToolUse

    uses = [
        ToolUse(id=f"toolu_{index:02d}_{name}", name=name, arguments=arguments)
        for index, (name, arguments) in enumerate(calls)
    ]
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(
        {"type": "tool_use", "id": use.id, "name": use.name, "input": use.arguments}
        for use in uses
    )
    return ModelResponse(
        text=text,
        tool_uses=uses,
        stop_reason="tool_use",
        raw_content=content,
        model_name=MODEL_NAME,
        latency_ms=latency_ms,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
    )


__all__ = [
    "INPUT_TOKENS",
    "MODEL_NAME",
    "OUTPUT_TOKENS",
    "ScriptExhausted",
    "ScriptedModel",
    "say",
    "use_tools",
]
