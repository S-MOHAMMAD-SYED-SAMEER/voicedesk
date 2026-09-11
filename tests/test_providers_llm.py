"""The model boundary: the interface, and the Anthropic implementation.

Nothing here needs an API key, a network or a database. The SDK client is
injected, so the request is built and the response parsed entirely offline.
"""

from typing import Any

import anthropic
import httpx2
import pytest

from app.config import Settings
from app.providers import (
    LanguageModel,
    Message,
    ModelError,
    ModelRefused,
    ModelResponse,
    ModelUnavailable,
    ToolDefinition,
)
from app.providers.anthropic_llm import AnthropicLanguageModel

MESSAGES = [Message(role="user", content=[{"type": "text", "text": "hello?"}])]
TOOLS = [
    ToolDefinition(
        name="check_availability",
        description="List free start times.",
        input_schema={
            "type": "object",
            "properties": {"day": {"type": "string"}},
            "required": ["day"],
            "additionalProperties": False,
        },
    )
]


class StubMessages:
    def __init__(self, response: Any, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.request = kwargs
        if self._error is not None:
            raise self._error
        return self._response


class StubClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.messages = StubMessages(response, error)


class StubUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class StubResponse:
    """Stands in for an SDK Message. Blocks are plain dicts."""

    def __init__(
        self,
        content: list[dict[str, Any]],
        stop_reason: str = "end_turn",
        usage: StubUsage | None = None,
        model: str = "claude-opus-5",
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage
        self.model = model


def _model(response: Any = None, error: Exception | None = None):
    client = StubClient(response, error)
    return (
        AnthropicLanguageModel(
            client=client,
            settings=Settings(_env_file=None, anthropic_api_key=""),
        ),
        client,
    )


def _response(status: int) -> httpx2.Response:
    return httpx2.Response(
        status,
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


# --- the interface itself -------------------------------------------------


def test_a_fake_satisfies_the_interface() -> None:
    """Anything with `respond(system, messages, tools)` is a language model."""

    class Fake:
        def respond(self, *, system, messages, tools):
            return ModelResponse(text="hello")

    model: LanguageModel = Fake()
    assert model.respond(system="s", messages=[], tools=[]).text == "hello"


def test_the_anthropic_provider_satisfies_the_interface() -> None:
    model, _ = _model(StubResponse([{"type": "text", "text": "hi"}]))
    checked: LanguageModel = model
    assert checked.respond(system="s", messages=MESSAGES, tools=TOOLS).text == "hi"


# --- request construction -------------------------------------------------


def test_the_request_carries_the_system_prompt_messages_and_tools() -> None:
    model, client = _model(StubResponse([{"type": "text", "text": "hi"}]))

    model.respond(system="be a receptionist", messages=MESSAGES, tools=TOOLS)

    request = client.messages.request
    assert request["system"] == "be a receptionist"
    assert request["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello?"}]}
    ]
    assert [tool["name"] for tool in request["tools"]] == ["check_availability"]
    assert request["tools"][0]["input_schema"] == TOOLS[0].input_schema


def test_tools_are_sent_as_strict_definitions() -> None:
    """Strict tool use guarantees the arguments validate against the schema."""
    model, client = _model(StubResponse([{"type": "text", "text": "hi"}]))

    model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert client.messages.request["tools"][0]["strict"] is True


def test_the_model_max_tokens_and_effort_come_from_settings() -> None:
    client = StubClient(StubResponse([{"type": "text", "text": "hi"}]))
    model = AnthropicLanguageModel(
        client=client,
        settings=Settings(
            _env_file=None,
            dialogue_model="claude-sonnet-5",
            dialogue_max_tokens=512,
            dialogue_effort="medium",
        ),
    )

    model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    request = client.messages.request
    assert request["model"] == "claude-sonnet-5"
    assert request["max_tokens"] == 512
    assert request["output_config"] == {"effort": "medium"}
    assert model.model_name == "claude-sonnet-5"


def test_thinking_is_adaptive_rather_than_a_token_budget() -> None:
    """The budgeted form is removed on current models."""
    model, client = _model(StubResponse([{"type": "text", "text": "hi"}]))

    model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert client.messages.request["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in str(client.messages.request)


def test_explicit_arguments_override_settings() -> None:
    client = StubClient(StubResponse([{"type": "text", "text": "hi"}]))
    model = AnthropicLanguageModel(
        client=client,
        model="claude-haiku-4-5",
        max_tokens=64,
        effort="high",
        settings=Settings(_env_file=None),
    )

    model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert client.messages.request["model"] == "claude-haiku-4-5"
    assert client.messages.request["max_tokens"] == 64
    assert client.messages.request["output_config"] == {"effort": "high"}


# --- response parsing -----------------------------------------------------


def test_a_text_response_is_read_back() -> None:
    model, _ = _model(StubResponse([{"type": "text", "text": "We're open till five."}]))

    response = model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert response.text == "We're open till five."
    assert response.tool_uses == []
    assert response.stop_reason == "end_turn"


def test_a_tool_use_response_is_read_back() -> None:
    model, _ = _model(
        StubResponse(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "check_availability",
                    "input": {"service_name": "Haircut", "day": "2026-03-02"},
                }
            ],
            stop_reason="tool_use",
        )
    )

    response = model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert len(response.tool_uses) == 1
    use = response.tool_uses[0]
    assert (use.id, use.name) == ("toolu_01", "check_availability")
    assert use.arguments == {"service_name": "Haircut", "day": "2026-03-02"}


def test_several_tool_calls_in_one_response_are_all_read() -> None:
    model, _ = _model(
        StubResponse(
            [
                {"type": "tool_use", "id": "a", "name": "cancel", "input": {}},
                {"type": "tool_use", "id": "b", "name": "take_message", "input": {}},
            ],
            stop_reason="tool_use",
        )
    )

    response = model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert [use.id for use in response.tool_uses] == ["a", "b"]


def test_text_alongside_tool_use_is_kept() -> None:
    model, _ = _model(
        StubResponse(
            [
                {"type": "text", "text": "Let me look."},
                {"type": "tool_use", "id": "a", "name": "cancel", "input": {}},
            ],
            stop_reason="tool_use",
        )
    )

    response = model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert response.text == "Let me look."
    assert len(response.tool_uses) == 1


def test_the_raw_blocks_are_kept_for_echoing_back() -> None:
    """Tool results only match their calls if the call blocks are returned."""
    blocks = [{"type": "tool_use", "id": "a", "name": "cancel", "input": {}}]
    model, _ = _model(StubResponse(list(blocks), stop_reason="tool_use"))

    response = model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert response.raw_content == blocks


def test_usage_and_latency_are_recorded() -> None:
    model, _ = _model(
        StubResponse([{"type": "text", "text": "hi"}], usage=StubUsage(1200, 48))
    )

    response = model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert (response.input_tokens, response.output_tokens) == (1200, 48)
    assert response.latency_ms is not None and response.latency_ms >= 0
    assert response.model_name == "claude-opus-5"


def test_missing_usage_is_none_rather_than_zero() -> None:
    """A guessed token count would become a guessed cost in milestone 8."""
    model, _ = _model(StubResponse([{"type": "text", "text": "hi"}]))

    response = model.respond(system="s", messages=MESSAGES, tools=TOOLS)

    assert response.input_tokens is None
    assert response.output_tokens is None


def test_sdk_content_objects_are_accepted_as_well_as_dicts() -> None:
    """The real client returns Pydantic blocks; both must read back the same."""

    class Block:
        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        def model_dump(self) -> dict[str, Any]:
            return self._body

    model, _ = _model(StubResponse([Block({"type": "text", "text": "hi"})]))

    assert model.respond(system="s", messages=MESSAGES, tools=TOOLS).text == "hi"


def test_an_unreadable_block_is_a_model_error() -> None:
    model, _ = _model(StubResponse([object()]))

    with pytest.raises(ModelError):
        model.respond(system="s", messages=MESSAGES, tools=TOOLS)


# --- error mapping --------------------------------------------------------


def test_a_refusal_is_raised_not_returned() -> None:
    """A refusal is not an answer; the dialogue layer must say something safe."""
    model, _ = _model(StubResponse([], stop_reason="refusal"))

    with pytest.raises(ModelRefused):
        model.respond(system="s", messages=MESSAGES, tools=TOOLS)


@pytest.mark.parametrize(
    "error",
    [
        anthropic.APITimeoutError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
        anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
        anthropic.RateLimitError("slow down", response=_response(429), body=None),
        anthropic.InternalServerError("boom", response=_response(500), body=None),
    ],
    ids=["timeout", "connection", "rate-limit", "server-error"],
)
def test_transport_failures_become_model_unavailable(error: Exception) -> None:
    model, _ = _model(error=error)

    with pytest.raises(ModelUnavailable):
        model.respond(system="s", messages=MESSAGES, tools=TOOLS)


def test_a_bad_request_is_a_model_error_but_not_unavailable() -> None:
    """Retrying a 400 would just fail again; it is our bug, not their outage."""
    model, _ = _model(
        error=anthropic.BadRequestError("bad", response=_response(400), body=None)
    )

    with pytest.raises(ModelError) as caught:
        model.respond(system="s", messages=MESSAGES, tools=TOOLS)
    assert not isinstance(caught.value, ModelUnavailable)


def test_no_api_key_is_needed_to_build_the_provider() -> None:
    """Every test here runs with an empty key and an injected client."""
    model, _ = _model(StubResponse([{"type": "text", "text": "hi"}]))

    assert model.respond(system="s", messages=MESSAGES, tools=TOOLS).text == "hi"
