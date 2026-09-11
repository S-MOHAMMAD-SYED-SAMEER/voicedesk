"""The audio layer: a microphone and a speaker around the dialogue layer.

    audio → STT → Conversation → TTS → audio

Nothing here contains dialogue logic, touches the calendar, runs a tool or
speaks to a model. It validates what comes in, hands text to milestone 4, and
turns the answer back into sound.

Milestone 5 is utterance-based: press to talk, release, hear the reply.
Streaming, voice-activity detection and barge-in belong to the milestone that
owns latency, and the 1.2-second target is neither met nor measured here.
"""

from app.audio.errors import AudioError, AudioTooLarge, InvalidAudio
from app.audio.format import (
    DEFAULT_MAX_BYTES,
    DEFAULT_SAMPLE_RATE,
    expected_format,
    validate_utterance,
)
from app.audio.session import (
    NOT_HEARD_REPLY,
    SPEECH_FAILURE_REPLY,
    VoiceSession,
    VoiceTurn,
    utterance_from_bytes,
)

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_SAMPLE_RATE",
    "NOT_HEARD_REPLY",
    "SPEECH_FAILURE_REPLY",
    "AudioError",
    "AudioTooLarge",
    "InvalidAudio",
    "VoiceSession",
    "VoiceTurn",
    "expected_format",
    "utterance_from_bytes",
    "validate_utterance",
]
