"""Provider interfaces: the language model, and speech in and out.

Only the vendor-neutral interfaces are re-exported here. Every implementation
lives in its own module and is imported explicitly by whoever wants it, so
importing this package never pulls in a vendor's SDK or request shapes. That
is what lets the dialogue layer, the audio layer and the whole test suite run
with no credentials and no network.
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
from app.providers.speech import Audio, AudioFormat, WavError, build_wav, read_wav
from app.providers.stt import (
    SpeechError,
    SpeechToText,
    SpeechUnavailable,
    Transcript,
    UnsupportedAudio,
)
from app.providers.tts import Speech, TextToSpeech, VoiceError, VoiceUnavailable

__all__ = [
    "Audio",
    "AudioFormat",
    "LanguageModel",
    "Message",
    "ModelError",
    "ModelRefused",
    "ModelResponse",
    "ModelUnavailable",
    "Speech",
    "SpeechError",
    "SpeechToText",
    "SpeechUnavailable",
    "TextToSpeech",
    "ToolDefinition",
    "ToolUse",
    "Transcript",
    "UnsupportedAudio",
    "VoiceError",
    "VoiceUnavailable",
    "WavError",
    "build_wav",
    "read_wav",
]
