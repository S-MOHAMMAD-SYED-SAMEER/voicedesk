"""Anthropic implementation of `LanguageModel`.

This is the only module in VoiceDesk that imports the Anthropic SDK. Nothing
else may — `tests/test_dialogue_isolation.py` enforces it — so the dialogue
layer, the tool layer and the whole test suite run with no API key and no
network.

The request is deliberately plain: a system prompt, the conversation so far,
the six tool definitions, and adaptive thinking with a configurable effort.
Streaming, prompt caching and cost accounting belong to the milestones that
need them.
"""

import time
from collections.abc import Sequence
from typing import Any

import anthropic

from app.config import Settings, get_settings
from app.providers.llm import (
    Message,
    ModelError,
    ModelRefused,
    ModelResponse,
    ModelUnavailable,
    ToolDefinition,
    ToolUse,
)


def _as_dict(block: Any) -> dict[str, Any]:
    """An SDK content block as a plain dict, so it can be echoed back.

    Blocks are Pydantic models on the real client; a test stub may hand back
    dicts directly. Both are accepted, and the result is what the next request
    sends verbatim.
    """
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return dump()
    raise ModelError(f"Cannot read a content block of type {type(block)!r}.")


class AnthropicLanguageModel:
    """Calls Claude with the conversation so far and the tools it may use."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._model = model or resolved.dialogue_model
        self._max_tokens = max_tokens or resolved.dialogue_max_tokens
        self._effort = effort or resolved.dialogue_effort
        # An explicit timeout, because the SDK's default is minutes and a
        # caller is on the telephone. It ends the *wait*: the request is
        # abandoned, and if it was running on a worker thread that thread
        # finishes into nothing. Nothing here can kill a thread.
        self._client = client or anthropic.Anthropic(
            api_key=resolved.anthropic_api_key or None,
            timeout=resolved.dialogue_timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def respond(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "strict": tool.strict,
                }
                for tool in tools
            ],
            # Adaptive thinking rather than a fixed token budget: the budgeted
            # form is removed on current models. Depth is controlled by effort,
            # which defaults low — a receptionist is on a latency budget.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
        }

        started = time.perf_counter()
        try:
            response = self._client.messages.create(**request)
        except anthropic.APITimeoutError as exc:
            raise ModelUnavailable(f"The model timed out: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ModelUnavailable(f"The model is rate limited: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise ModelUnavailable(f"The model could not be reached: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ModelUnavailable(
                    f"The model returned {exc.status_code}: {exc}"
                ) from exc
            raise ModelError(f"The model rejected the request: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        if stop_reason == "refusal":
            raise ModelRefused("The model declined to answer.")

        blocks = [_as_dict(block) for block in response.content]
        text = "\n".join(
            block["text"] for block in blocks if block.get("type") == "text"
        ).strip()
        tool_uses = [
            ToolUse(id=block["id"], name=block["name"], arguments=block["input"])
            for block in blocks
            if block.get("type") == "tool_use"
        ]

        usage = getattr(response, "usage", None)
        return ModelResponse(
            text=text,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            raw_content=blocks,
            model_name=getattr(response, "model", self._model) or self._model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            latency_ms=latency_ms,
        )
