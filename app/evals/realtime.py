"""Driving a realtime call, so barge-in and silence can be evaluated.

Two scenarios need the streaming path rather than the text path, and neither
can be faked at the dialogue layer:

* **Silence** is refused above `Conversation`. `Conversation.send("")` would
  still ask the model; it is `RealtimeSession` that declines to. So proving
  "silence asks no model" means feeding real frames through the real
  voice-activity detector.
* **Barge-in** is entirely a property of `RealtimeSession`'s generation model.

Nothing in M7 is modified or worked around. The one piece of apparatus here is
`PacedTextToSpeech`: a streaming synthesiser that holds after its first chunk
until it is released. Without it the whole reply is delivered inside a single
scheduling slice and there is no "mid-reply" to interrupt — the interruption
would be measured against a call that had already finished speaking. It is a
clock, not a stand-in for behaviour, and it implements the ordinary
`StreamingTextToSpeech` protocol.
"""

import math
import struct

import anyio

from app.providers.speech import PCM_S16LE, AudioFormat
from app.providers.streaming_tts import SpeechChunk

# One 20 ms frame at the telephony rate, which is the grain everything works
# in. Loud enough to read as speech; silence is silence.
SAMPLE_RATE = 8000
FRAME_MS = 20
SPEECH_AMPLITUDE = 9000
TONE_HZ = 220.0

# How long to let a held reply sit before interrupting it. Long enough for the
# turn task to have started speaking, short enough that 18 scenarios stay fast.
SETTLE_SECONDS = 0.05


def speech_frame(ms: int = FRAME_MS) -> bytes:
    """One frame of a tone: what "somebody is talking" sounds like."""
    count = int(SAMPLE_RATE * ms / 1000)
    return struct.pack(
        f"<{count}h",
        *(
            int(SPEECH_AMPLITUDE * math.sin(2 * math.pi * TONE_HZ * i / SAMPLE_RATE))
            for i in range(count)
        ),
    )


def silent_frame(ms: int = FRAME_MS) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * ms / 1000)


class PacedVoiceStream:
    """A reply delivered a chunk at a time, pausing after the first.

    The pause is the whole point: it leaves the session demonstrably mid-reply
    so that an interruption has something to interrupt.
    """

    CHUNKS = 5

    def __init__(self, text: str, audio_format: AudioFormat, gate: anyio.Event) -> None:
        self._text = text
        self._format = audio_format
        self._gate = gate
        self.closed = False
        self.produced = 0

    async def chunks(self):
        for index in range(self.CHUNKS):
            if self.closed:
                return
            self.produced += 1
            yield SpeechChunk(
                audio=b"\x01\x02" * 80,
                format=self._format,
                is_final=index == self.CHUNKS - 1,
                characters=len(self._text),
                metadata={"provider": "offline"},
            )
            if index == 0:
                await self._gate.wait()

    async def aclose(self) -> None:
        self.closed = True
        # Releasing on close keeps a held generator from outliving its call.
        self._gate.set()


class PacedTextToSpeech:
    """Opens `PacedVoiceStream`s, one per reply."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._format = AudioFormat(PCM_S16LE, sample_rate, 1, "raw")
        self.gate = anyio.Event()
        self.streams: list[PacedVoiceStream] = []
        self.spoken: list[str] = []

    @property
    def format(self) -> AudioFormat:
        return self._format

    def stream(self, text: str, voice: str | None = None) -> PacedVoiceStream:
        self.spoken.append(text)
        opened = PacedVoiceStream(text, self._format, self.gate)
        self.streams.append(opened)
        return opened


async def feed(session, frames: int, frame: bytes) -> None:
    """Hand the session `frames` copies of one frame."""
    for _ in range(frames):
        await session.feed(frame)


__all__ = [
    "FRAME_MS",
    "SAMPLE_RATE",
    "SETTLE_SECONDS",
    "PacedTextToSpeech",
    "PacedVoiceStream",
    "feed",
    "silent_frame",
    "speech_frame",
]
