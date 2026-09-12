"""Where a reply's audio goes, whoever is carrying it.

Three operations, and each exists because barge-in needs it:

* `send` writes a chunk toward the caller,
* `clear` discards whatever is queued but not yet heard,
* `mark` asks to be told when what has been sent has finished playing.

The session speaks only in these terms. A carrier's JSON and a browser's
frames are implemented where those transports live, not here, and nothing in
this module knows which one it is talking to.
"""

from typing import Protocol

from app.providers.streaming_tts import SpeechChunk


class AudioSink(Protocol):
    """A place to put audio that the caller will hear."""

    async def send(self, chunk: SpeechChunk) -> None: ...

    async def clear(self) -> None:
        """Discard queued audio. Called the moment somebody interrupts."""
        ...

    async def mark(self, name: str) -> None:
        """Note the end of a batch, so playback completion can be observed."""
        ...


class NullSink:
    """Accepts audio and drops it. For tests that are not about playback."""

    def __init__(self) -> None:
        self.chunks: list[SpeechChunk] = []
        self.clears = 0
        self.marks: list[str] = []

    async def send(self, chunk: SpeechChunk) -> None:
        self.chunks.append(chunk)

    async def clear(self) -> None:
        self.chunks.clear()
        self.clears += 1

    async def mark(self, name: str) -> None:
        self.marks.append(name)
