"""The realtime layer: a conversation that can be interrupted.

    frames → VAD → endpointer → streaming STT → Conversation → streaming TTS → sink

The streaming counterpart to `app/audio/session.py`, which still serves
press-to-talk and is unchanged. Both reach the dialogue through
`Conversation`, so there is one dialogue implementation, one tool registry and
one calendar beneath them.

Nothing here knows about carriers, JSON, µ-law, sockets, SQL, tools, the
calendar, or any vendor.
"""

from app.realtime.generation import Generation, Generations
from app.realtime.latency import record as record_latency
from app.realtime.session import (
    NOT_HEARD_REPLY,
    SPEECH_FAILURE_REPLY,
    RealtimeSession,
    RealtimeTurn,
    TurnTiming,
)
from app.realtime.sink import AudioSink, NullSink

__all__ = [
    "NOT_HEARD_REPLY",
    "SPEECH_FAILURE_REPLY",
    "AudioSink",
    "Generation",
    "Generations",
    "NullSink",
    "record_latency",
    "RealtimeSession",
    "RealtimeTurn",
    "TurnTiming",
]
