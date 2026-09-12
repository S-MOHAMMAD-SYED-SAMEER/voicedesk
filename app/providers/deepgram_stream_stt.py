"""Deepgram's streaming recognition, over a plain WebSocket.

The only module that knows Deepgram's realtime protocol. No vendor SDK: the
service is a WebSocket that takes raw audio as binary frames and answers with
JSON, so the `websockets` client already in this project is the whole
dependency, and the connection is injectable — which is how the tests exercise
the protocol without a key or a network.

**Written from documented protocol knowledge and never run against the real
service from this repository.** The frame shapes and the close handshake are
what the tests verify; recognition quality, latency and reconnect behaviour
are not claimed and have not been observed. If a field name here disagrees
with Deepgram's current documentation, believe the documentation.

On a mid-utterance disconnect this raises rather than reconnecting. A
reconnect loses the audio the service had already buffered, and a silently
truncated transcript is worse than the caller being asked to repeat
themselves.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import anyio

from app.config import Settings, get_settings
from app.providers.speech import AudioFormat
from app.providers.streaming import messages
from app.providers.streaming_stt import (
    FinalTranscript,
    PartialTranscript,
    SpeechError,
    SpeechUnavailable,
    TranscriptEvent,
    UnsupportedAudio,
)

logger = logging.getLogger(__name__)

ENDPOINT = "wss://api.deepgram.com/v1/listen"
PROVIDER_NAME = "deepgram"
# What the audio layer produces. Sent as a query parameter because the stream
# carries headerless samples, so the service is told out of band.
LINEAR16 = "linear16"
CLOSE_MESSAGE = {"type": "CloseStream"}


def connection_url(audio_format: AudioFormat, model: str) -> str:
    """The endpoint, with this stream's audio described in the query."""
    if audio_format.encoding != "pcm_s16le":
        raise UnsupportedAudio(
            f"This adapter sends 16-bit PCM; it was handed "
            f"{audio_format.encoding!r}."
        )
    query = urlencode(
        {
            "model": model,
            "encoding": LINEAR16,
            "sample_rate": audio_format.sample_rate,
            "channels": audio_format.channels,
            "interim_results": "true",
            "smart_format": "true",
        }
    )
    return f"{ENDPOINT}?{query}"


class DeepgramSpeechStream:
    """One utterance's recognition, over one socket."""

    def __init__(
        self,
        connection: Any,
        audio_format: AudioFormat,
        idle_timeout: float = 20.0,
    ) -> None:
        self._connection = connection
        self._format = audio_format
        self._idle_timeout = idle_timeout
        self._closed = False

    async def send(self, pcm: bytes) -> None:
        if self._closed:
            raise SpeechError("This recognition stream is closed.")
        try:
            await self._connection.send(pcm)
        except Exception as exc:  # noqa: BLE001 - any transport failure
            raise SpeechUnavailable(f"Deepgram could not be sent audio: {exc}") from exc

    async def finish(self) -> None:
        """Ask for whatever is left. The service answers, then closes."""
        if self._closed:
            return
        try:
            await self._connection.send(json.dumps(CLOSE_MESSAGE))
        except Exception as exc:  # noqa: BLE001
            raise SpeechUnavailable(
                f"Deepgram could not be asked to finish: {exc}"
            ) from exc

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        """Interim results, then the one that counts."""
        try:
            async for raw in messages(self._connection, self._idle_timeout):
                event = _read(raw, self._format)
                if event is not None:
                    yield event
                    if isinstance(event, FinalTranscript):
                        return
        except SpeechError:
            raise
        except Exception as exc:  # noqa: BLE001
            # A disconnect mid-utterance. Not reconnected, by design.
            raise SpeechUnavailable(f"Deepgram stopped responding: {exc}") from exc

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._connection.close()
        except Exception:  # noqa: BLE001 - closing must not raise
            logger.debug("Deepgram socket did not close cleanly.", exc_info=True)


def _read(raw: str | bytes, audio_format: AudioFormat) -> TranscriptEvent | None:
    """One message from the service, or nothing if it said nothing useful."""
    try:
        body: dict[str, Any] = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SpeechError("Deepgram sent something that is not JSON.") from exc
    if not isinstance(body, dict):
        raise SpeechError("Deepgram sent something that is not a JSON object.")

    kind = body.get("type")
    if kind in {"Metadata", "SpeechStarted", "UtteranceEnd"}:
        return None
    if kind == "Error":
        raise SpeechError(f"Deepgram reported an error: {body.get('description')}")

    channel = body.get("channel")
    if not isinstance(channel, dict):
        return None
    alternatives = channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return None
    best = alternatives[0]
    if not isinstance(best, dict):
        return None

    text = best.get("transcript")
    if not isinstance(text, str):
        return None
    confidence = best.get("confidence")
    confidence = confidence if isinstance(confidence, int | float) else None

    # `speech_final` means the service believes the utterance is over;
    # `is_final` alone only means this segment will not be revised.
    if body.get("speech_final") or body.get("from_finalize"):
        duration = body.get("duration")
        return FinalTranscript(
            text=text.strip(),
            confidence=confidence,
            language=channel.get("detected_language"),
            audio_ms=(
                round(duration * 1000) if isinstance(duration, int | float) else None
            ),
            provider_name=PROVIDER_NAME,
            metadata={"is_final": bool(body.get("is_final"))},
        )
    if not text.strip():
        return None
    return PartialTranscript(
        text=text.strip(),
        confidence=confidence,
        at_ms=round((body.get("start") or 0) * 1000),
    )


class DeepgramStreamingSpeechToText:
    """Opens recognition sockets. One per utterance."""

    def __init__(
        self,
        connect=None,
        api_key: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._api_key = api_key if api_key is not None else resolved.deepgram_api_key
        self._model = model or resolved.stt_model
        self._connect = connect
        self._connect_timeout = resolved.stream_connect_timeout_seconds
        self._read_timeout = resolved.stream_read_timeout_seconds

    def stream(self, audio_format: AudioFormat) -> "_PendingStream":
        if not self._api_key:
            raise SpeechUnavailable("No Deepgram API key is configured.")
        return _PendingStream(
            url=connection_url(audio_format, self._model),
            headers={"Authorization": f"Token {self._api_key}"},
            audio_format=audio_format,
            connect=self._connect,
            connect_timeout=self._connect_timeout,
            read_timeout=self._read_timeout,
        )


class _PendingStream:
    """A stream that opens its socket on first use.

    `stream()` is not a coroutine — the interface is deliberately synchronous
    so a caller can hold one before awaiting anything — so the connection is
    made when audio first arrives.
    """

    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        audio_format,
        connect,
        connect_timeout: float = 10.0,
        read_timeout: float = 20.0,
    ):
        self._url = url
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._headers = headers
        self._format = audio_format
        self._connect = connect
        self._opened: DeepgramSpeechStream | None = None

    async def _stream(self) -> DeepgramSpeechStream:
        if self._opened is None:
            connect = self._connect
            if connect is None:  # pragma: no cover - needs a real network
                import websockets

                connect = websockets.connect
            try:
                with anyio.fail_after(self._connect_timeout):
                    connection = await connect(
                        self._url, additional_headers=self._headers
                    )
            except Exception as exc:  # noqa: BLE001
                raise SpeechUnavailable(
                    f"Deepgram could not be reached: {exc}"
                ) from exc
            self._opened = DeepgramSpeechStream(
                connection, self._format, self._read_timeout
            )
        return self._opened

    async def send(self, pcm: bytes) -> None:
        await (await self._stream()).send(pcm)

    async def finish(self) -> None:
        await (await self._stream()).finish()

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        stream = await self._stream()
        async for event in stream.events():
            yield event

    async def aclose(self) -> None:
        if self._opened is not None:
            await self._opened.aclose()
