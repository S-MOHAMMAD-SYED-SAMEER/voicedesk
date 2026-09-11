"""The model-call boundary.

Everything above this line works in terms of `respond(system, messages, tools)`
and `ModelResponse`. Nothing above it imports a vendor SDK, so swapping or
adding a provider touches only `app/providers/`.

The specification's layout reserves this package for `stt.py`, `tts.py` and
`llm.py`; speech arrives with the milestones that need audio. Milestone 4
needs only the language model.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolUse:
    """One tool the model asked for.

    `id` is the provider's own correlation identifier. It matters only while a
    request is in flight — a result has to be sent back carrying the same id —
    so it lives here and is never written to the database.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Message:
    """One entry of conversation history, in provider-neutral block form.

    `content` is a list of blocks (text, tool use, tool result) rather than a
    string, because a turn that used tools is not expressible as one.
    """

    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    """A tool as the model sees it: a name, a description and a schema.

    `strict` asks the provider to guarantee the arguments validate against
    `input_schema`. It is a belt to the tool layer's braces — the tools
    themselves still normalise and refuse what they are given.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True)
class ModelResponse:
    """What the model said, and what it wants to do about it.

    `raw_content` is the response's own blocks, kept verbatim so the next
    request can echo them back unchanged — which is how tool results are
    matched to the calls that asked for them.
    """

    text: str
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    model_name: str = ""
    # Recorded for milestone 8's cost tracking. Milestone 4 stores no cost.
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None


class LanguageModel(Protocol):
    """The one call the dialogue layer makes to a model."""

    def respond(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse: ...


class ModelError(Exception):
    """The model could not produce an answer.

    Raised rather than returned, unlike a tool failure. A tool that refuses is
    information the model can act on; a model that cannot answer ends the turn,
    and the dialogue layer has to say something safe instead of inventing one.
    """


class ModelUnavailable(ModelError):
    """The provider could not be reached, or would not serve the request."""


class ModelRefused(ModelError):
    """The model declined to answer."""
