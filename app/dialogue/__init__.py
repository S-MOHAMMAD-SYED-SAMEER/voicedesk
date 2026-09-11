"""The dialogue layer: a conversation with a model that can use the tools.

    caller text → Conversation → LanguageModel → tool_use
                                      ↓
                               ToolExecutor → app.tools registry
                                      ↓
                             CalendarService → PostgreSQL
                                      ↓
                              tool_result → LanguageModel → reply

Text in, text out. Nothing here knows about audio, telephony or a phone
number, and nothing here talks to the calendar or the database except through
the milestone-3 tools and the transcript it writes.
"""

from app.dialogue.conversation import (
    LOOP_EXHAUSTED_REPLY,
    MODEL_FAILURE_REPLY,
    Conversation,
    DialogueResult,
)
from app.dialogue.definitions import tool_definitions
from app.dialogue.errors import DialogueError, ToolLoopExhausted
from app.dialogue.executor import UNVERIFIED_SLOT, ToolCallRecord, ToolExecutor
from app.dialogue.prompt import SYSTEM_PROMPT_VERSION, build_system_prompt

__all__ = [
    "LOOP_EXHAUSTED_REPLY",
    "MODEL_FAILURE_REPLY",
    "SYSTEM_PROMPT_VERSION",
    "UNVERIFIED_SLOT",
    "Conversation",
    "DialogueError",
    "DialogueResult",
    "ToolCallRecord",
    "ToolExecutor",
    "ToolLoopExhausted",
    "build_system_prompt",
    "tool_definitions",
]
