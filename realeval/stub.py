"""An offline model for checking the machinery. It measures nothing.

`python -m realeval` with no `--provider` runs every scenario against this: it answers every request with one fixed
sentence and never calls a tool. Almost every scenario therefore *fails* (nothing is booked, nobody is escalated), which
is the point: it shows the runner, the database, the checks and the reports work end to end with no key and no network,
and that a model that does nothing does not pass.
"""

from collections.abc import Sequence

from app.providers.llm import Message, ModelResponse, ToolDefinition

REPLY = "I'm sorry, I can't help with that."


class StubModel:
    model_name = "stub"

    def __init__(self) -> None:
        self.call_count = 0
        self.halt_reason: str | None = None

    def respond(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(
            text=REPLY,
            stop_reason="end_turn",
            raw_content=[{"type": "text", "text": REPLY}],
            model_name="stub",
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
        )
