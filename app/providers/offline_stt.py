"""A transcriber that does no speech recognition at all.

This is **not** speech recognition, and nothing in this project will pretend
it is. It reads the audio's header, checks the shape is one the system
accepts, and returns a fixed line. Its purpose is that a fresh clone can run
the whole browser harness — microphone, dialogue, speaker — with no API key,
no network and no bill, and that the test suite has something deterministic to
put in front of the dialogue layer.

Whatever you say into the microphone, it hears the same sentence.
"""

from app.providers.speech import Audio, WavError, read_wav
from app.providers.stt import Transcript, UnsupportedAudio

DEFAULT_TRANSCRIPT = "I'd like to book an appointment"
PROVIDER_NAME = "offline"


class OfflineSpeechToText:
    """Returns a fixed transcript for any acceptable audio.

    `transcripts` lets a test script a conversation: each call takes the next
    one, and the last is repeated once the script runs out, so a test never
    fails for the uninteresting reason of having run dry.
    """

    def __init__(
        self,
        text: str = DEFAULT_TRANSCRIPT,
        *,
        transcripts: list[str] | None = None,
        confidence: float | None = 1.0,
        language: str | None = "en",
    ) -> None:
        self._scripted = list(transcripts) if transcripts else [text]
        self._confidence = confidence
        self._language = language
        self.calls = 0

    def transcribe(self, audio: Audio) -> Transcript:
        try:
            info = read_wav(audio.data)
        except WavError as exc:
            raise UnsupportedAudio(str(exc)) from exc

        index = min(self.calls, len(self._scripted) - 1)
        self.calls += 1
        return Transcript(
            text=self._scripted[index],
            confidence=self._confidence,
            language=self._language,
            audio_ms=info.duration_ms,
            latency_ms=0,
            provider_name=PROVIDER_NAME,
            metadata={"offline": True},
        )
