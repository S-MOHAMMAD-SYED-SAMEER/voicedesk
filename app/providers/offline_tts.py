"""A synthesiser that produces a tone, not a voice.

Honest about what it is: it generates a short sine tone whose length is
derived from the text, so the harness has something real to play and the
tests have deterministic bytes to assert on. Nobody will mistake it for
speech, and the README says so.

It exists for the same reason the offline transcriber does — a clone with no
credentials can still run the whole path end to end.
"""

import math
import struct

from app.providers.speech import PCM_S16LE, AudioFormat, build_wav
from app.providers.tts import Speech

PROVIDER_NAME = "offline"
DEFAULT_VOICE = "offline-tone"
# Roughly a speaking pace, so a long reply plays for longer than a short one.
MS_PER_CHARACTER = 60
MIN_DURATION_MS = 200
MAX_DURATION_MS = 20_000
TONE_HZ = 220.0
AMPLITUDE = 8000


class OfflineTextToSpeech:
    """Turns text into a tone of a proportionate length."""

    def __init__(self, sample_rate: int = 16000, voice: str = DEFAULT_VOICE) -> None:
        self._format = AudioFormat(
            encoding=PCM_S16LE, sample_rate=sample_rate, channels=1, container="wav"
        )
        self._voice = voice
        self.calls: list[str] = []

    @property
    def format(self) -> AudioFormat:
        return self._format

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        self.calls.append(text)
        duration_ms = min(
            MAX_DURATION_MS, max(MIN_DURATION_MS, len(text) * MS_PER_CHARACTER)
        )
        samples = round(self._format.sample_rate * duration_ms / 1000)
        step = 2 * math.pi * TONE_HZ / self._format.sample_rate
        pcm = struct.pack(
            f"<{samples}h",
            *(int(AMPLITUDE * math.sin(step * index)) for index in range(samples)),
        )

        return Speech(
            audio=build_wav(pcm, self._format),
            format=self._format,
            duration_ms=duration_ms,
            voice=voice or self._voice,
            latency_ms=0,
            provider_name=PROVIDER_NAME,
            characters=len(text),
            metadata={"offline": True},
        )
