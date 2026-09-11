"""Errors the dialogue layer raises for itself.

Model failures come from `app.providers.llm` and keep their own names; these
are the two things that can go wrong in the loop rather than in the model.
"""


class DialogueError(Exception):
    """Base for everything this package raises."""


class ToolLoopExhausted(DialogueError):
    """One caller turn used up its tool-call budget without finishing.

    Caught by `Conversation`, which answers with a fixed safe line. Letting it
    escape, or letting the loop run on, would both be worse than saying so.
    """
