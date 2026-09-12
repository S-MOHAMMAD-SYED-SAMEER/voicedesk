"""Streaming providers that do no recognition and produce no voice.

The same bargain as the batch offline providers, and the same honesty about
it: this transcriber returns a fixed sentence whatever is said to it, and
this synthesiser returns a tone. They exist so a clone with no account can run
a realtime call end to end, and so every test in this milestone — partials,
finals, cancellation, barge-in, latency — is deterministic and reaches no
network.

Neither is speech recognition and neither is a voice.
"""

import asyncio
import math
import struct
from collections.abc import AsyncIterator

from app.providers.speech import PCM_S16LE, AudioFormat
from app.providers.streaming_stt import (
    FinalTranscript,
    PartialTranscript,
    SpeechError,
    TranscriptEvent,
)
from app.providers.streaming_tts import SpeechChunk

PROVIDER_NAME = "offline"
DEFAULT_TRANSCRIPT = "I'd like to book an appointment"
# What the fixed sentence looks like while it is still being "heard".
DEFAULT_PARTIALS = ("I'd like", "I'd like to book")
DEFAULT_VOICE = "offline-tone"
TONE_HZ = 220.0
AMPLITUDE = 8000
MS_PER_CHARACTER = 60
MIN_DURATION_MS = 200
MAX_DURATION_MS = 20_000
CHUNK_MS = 200


class OfflineSpeechStream:
    """Emits scripted partials, then one final, whatever it was sent."""

    def __init__(
        self,
        audio_format: AudioFormat,
        *,
        partials: tuple[str, ...] = DEFAULT_PARTIALS,
        final: str = DEFAULT_TRANSCRIPT,
        confidence: float | None = 1.0,
        raises: Exception | None = None,
    ) -> None:
        self._format = audio_format
        self._partials = partials
        self._final = final
        self._confidence = confidence
        self._raises = raises
        self._received = bytearray()
        self._finished = asyncio.Event()
        self._closed = False

    @property
    def received_ms(self) -> float:
        rate = self._format.sample_rate or 1
        return len(self._received) / 2 * 1000 / rate

    async def send(self, pcm: bytes) -> None:
        if self._closed:
            raise SpeechError("This recognition stream is closed.")
        self._received.extend(pcm)

    async def finish(self) -> None:
        self._finished.set()

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        if self._raises is not None:
            raise self._raises

        for index, text in enumerate(self._partials, start=1):
            if self._closed:
                return
            yield PartialTranscript(
                text=text, confidence=self._confidence, at_ms=index * 100
            )

        # A real provider would keep listening; this one waits to be told the
        # utterance is over, so a test controls exactly when the final lands.
        await self._finished.wait()
        if self._closed:
            return
        yield FinalTranscript(
            text=self._final,
            confidence=self._confidence,
            language="en",
            audio_ms=round(self.received_ms),
            provider_name=PROVIDER_NAME,
            metadata={"offline": True},
        )

    async def aclose(self) -> None:
        self._closed = True
        self._finished.set()


class OfflineStreamingSpeechToText:
    """Opens `OfflineSpeechStream`s. Injectable for tests."""

    def __init__(
        self,
        *,
        partials: tuple[str, ...] = DEFAULT_PARTIALS,
        final: str = DEFAULT_TRANSCRIPT,
        confidence: float | None = 1.0,
        raises: Exception | None = None,
    ) -> None:
        self._partials = partials
        self._final = final
        self._confidence = confidence
        self._raises = raises
        self.streams: list[OfflineSpeechStream] = []

    def stream(self, audio_format: AudioFormat) -> OfflineSpeechStream:
        opened = OfflineSpeechStream(
            audio_format,
            partials=self._partials,
            final=self._final,
            confidence=self._confidence,
            raises=self._raises,
        )
        self.streams.append(opened)
        return opened


class OfflineVoiceStream:
    """A tone of a length proportionate to the text, in fixed-size chunks."""

    def __init__(
        self,
        text: str,
        audio_format: AudioFormat,
        *,
        chunk_ms: int = CHUNK_MS,
        ms_per_character: int = MS_PER_CHARACTER,
        raises: Exception | None = None,
    ) -> None:
        self._text = text
        self._format = audio_format
        self._chunk_ms = chunk_ms
        self._ms_per_character = ms_per_character
        self._raises = raises
        self._closed = False

    async def chunks(self) -> AsyncIterator[SpeechChunk]:
        if self._raises is not None:
            raise self._raises

        rate = self._format.sample_rate
        total_ms = min(
            MAX_DURATION_MS,
            max(MIN_DURATION_MS, len(self._text) * self._ms_per_character),
        )
        per_chunk = round(rate * self._chunk_ms / 1000)
        total_samples = round(rate * total_ms / 1000)
        step = 2 * math.pi * TONE_HZ / rate

        offset = 0
        while offset < total_samples:
            if self._closed:
                return
            count = min(per_chunk, total_samples - offset)
            pcm = struct.pack(
                f"<{count}h",
                *(
                    int(AMPLITUDE * math.sin(step * (offset + index)))
                    for index in range(count)
                ),
            )
            offset += count
            yield SpeechChunk(
                audio=pcm,
                format=self._format,
                is_final=offset >= total_samples,
                characters=len(self._text),
                metadata={"offline": True},
            )

    async def aclose(self) -> None:
        self._closed = True


class OfflineStreamingTextToSpeech:
    """Opens `OfflineVoiceStream`s."""

    def __init__(
        self,
        sample_rate: int = 16000,
        *,
        chunk_ms: int = CHUNK_MS,
        # Shortened by the tests, so a suite is not spent listening to a tone
        # played at the speed a caller would hear it.
        ms_per_character: int = MS_PER_CHARACTER,
        raises: Exception | None = None,
    ) -> None:
        self._format = AudioFormat(
            encoding=PCM_S16LE, sample_rate=sample_rate, channels=1, container="raw"
        )
        self._chunk_ms = chunk_ms
        self._ms_per_character = ms_per_character
        self._raises = raises
        self.spoken: list[str] = []

    @property
    def format(self) -> AudioFormat:
        return self._format

    def stream(self, text: str, voice: str | None = None) -> OfflineVoiceStream:
        self.spoken.append(text)
        return OfflineVoiceStream(
            text,
            self._format,
            chunk_ms=self._chunk_ms,
            ms_per_character=self._ms_per_character,
            raises=self._raises,
        )
