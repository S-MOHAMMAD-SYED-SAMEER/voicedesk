"""Provider interfaces.

Only the vendor-neutral interface is re-exported here. The Anthropic
implementation lives in `app.providers.anthropic_llm` and is imported
explicitly by whoever wants it, so importing this package never pulls in an
SDK — which is what lets the dialogue layer and its tests run without one.
"""

from app.providers.llm import (
    LanguageModel,
    Message,
    ModelError,
    ModelRefused,
    ModelResponse,
    ModelUnavailable,
    ToolDefinition,
    ToolUse,
)

__all__ = [
    "LanguageModel",
    "Message",
    "ModelError",
    "ModelRefused",
    "ModelResponse",
    "ModelUnavailable",
    "ToolDefinition",
    "ToolUse",
]
