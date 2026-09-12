"""Which turn a piece of work belongs to, and whether it is still wanted.

A caller who interrupts makes everything in flight obsolete, and some of it
cannot be stopped: a thread running a model request and a database write
cannot be killed, only abandoned. So the question is never "did the work
stop?" — it is "does this result still belong to the conversation?".

Every turn takes a number. The session holds the current one. Work carries the
number it started under, and anything arriving under an older number is
dropped at the point of use rather than trusted because it arrived.

Checked in four places, deliberately more than once:

* before a final transcript reaches the dialogue layer,
* before synthesis is started for a reply,
* before any chunk of audio is written to the caller,
* after a cancelled task returns anyway.

The last is the one that matters most. A provider can have chunks in flight
when it is closed, and a thread can finish after it is abandoned.
"""

from dataclasses import dataclass


class Generations:
    """A monotonic counter, one per call."""

    def __init__(self) -> None:
        self._current = 0

    @property
    def current(self) -> int:
        return self._current

    def next(self) -> int:
        """Start a new turn, making everything older stale."""
        self._current += 1
        return self._current

    def is_current(self, generation: int) -> bool:
        return generation == self._current

    def token(self, generation: int | None = None) -> "Generation":
        """A handle the rest of the code can carry and ask."""
        return Generation(
            self, self._current if generation is None else generation
        )


@dataclass(frozen=True)
class Generation:
    """One turn's claim on the conversation."""

    generations: Generations
    number: int

    @property
    def current(self) -> bool:
        return self.generations.is_current(self.number)

    @property
    def stale(self) -> bool:
        return not self.current
