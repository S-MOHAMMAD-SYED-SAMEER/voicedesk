"""ElevenLabs' streaming synthesis, over a plain WebSocket.

The only module that knows ElevenLabs' realtime protocol. No vendor SDK: the
service takes JSON messages carrying text and answers with JSON carrying
base64 audio, so the `websockets` client already in this project is the whole
dependency, and the connection is injectable.

**Written from documented protocol knowledge and never run against the real
service from this repository.** The message shapes are what the tests verify;
voice quality, latency and the service's own chunking are not claimed and have
not been observed. If a field name here disagrees with ElevenLabs' current
documentation, believe the documentation.

Audio is requested as headerless PCM at the configured rate, because that is
what the streaming interface carries and what both transports want: a WAV
header per chunk would be a container the caller cannot chain.
"""

import base64
import binascii
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import anyio

from app.config import Settings, get_settings
from app.providers.speech import PCM_S16LE, AudioFormat
from app.providers.streaming import messages
from app.providers.streaming_tts import SpeechChunk, VoiceError, VoiceUnavailable

logger = logging.getLogger(__name__)

ENDPOINT = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
PROVIDER_NAME = "elevenlabs"
# The service names raw PCM output by its rate.
OUTPUT_FORMATS = {16000: "pcm_16000", 22050: "pcm_22050", 24000: "pcm_24000"}
# An empty text message is how the protocol says "that is the whole sentence".
END_OF_TEXT: dict[str, Any] = {"text": ""}


def connection_url(voice: str, model: str, output_format: str) -> str:
    query = {"output_format": output_format}
    if model:
        query["model_id"] = model
    return f"{ENDPOINT.format(voice_id=voice)}?{urlencode(query)}"


class ElevenLabsVoiceStream:
    """One reply being spoken, over one socket."""

    def __init__(
        self,
        *,
        text: str,
        url: str,
        api_key: str,
        audio_format: AudioFormat,
        connect,
        connect_timeout: float = 10.0,
        read_timeout: float = 20.0,
    ) -> None:
        self._text = text
        self._url = url
        self._api_key = api_key
        self._format = audio_format
        self._connect = connect
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._connection = None
        self._closed = False

    async def chunks(self) -> AsyncIterator[SpeechChunk]:
        connect = self._connect
        if connect is None:  # pragma: no cover - needs a real network
            import websockets

            connect = websockets.connect

        try:
            with anyio.fail_after(self._connect_timeout):
                self._connection = await connect(self._url)
        except Exception as exc:  # noqa: BLE001
            raise VoiceUnavailable(f"ElevenLabs could not be reached: {exc}") from exc

        try:
            # The first message carries the credential and the whole sentence;
            # the second says there is no more of it. The reply text is
            # already complete — this milestone streams the audio, not the
            # model — so there is nothing to send incrementally.
            await self._connection.send(
                json.dumps({"text": self._text, "xi_api_key": self._api_key})
            )
            await self._connection.send(json.dumps(END_OF_TEXT))

            async for raw in messages(self._connection, self._read_timeout):
                if self._closed:
                    return
                chunk = self._read(raw)
                if chunk is not None:
                    yield chunk
                    if chunk.is_final:
                        return
        except VoiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VoiceUnavailable(f"ElevenLabs stopped responding: {exc}") from exc

    def _read(self, raw: str | bytes) -> SpeechChunk | None:
        try:
            body: dict[str, Any] = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise VoiceError("ElevenLabs sent something that is not JSON.") from exc
        if not isinstance(body, dict):
            raise VoiceError("ElevenLabs sent something that is not a JSON object.")

        if body.get("error") or body.get("message") and body.get("code"):
            raise VoiceError(f"ElevenLabs reported an error: {body.get('message')}")

        payload = body.get("audio")
        is_final = bool(body.get("isFinal"))
        if not payload:
            # A final message may carry no audio; anything else empty is a
            # keep-alive and is not a chunk.
            if is_final:
                return SpeechChunk(
                    audio=b"",
                    format=self._format,
                    is_final=True,
                    characters=len(self._text),
                    metadata={"provider": PROVIDER_NAME},
                )
            return None

        try:
            audio = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise VoiceError("ElevenLabs sent audio that is not valid base64.") from exc
        if len(audio) % 2:
            raise VoiceError(
                "ElevenLabs sent an odd number of bytes, which cannot be "
                "16-bit samples."
            )

        return SpeechChunk(
            audio=audio,
            format=self._format,
            is_final=is_final,
            characters=len(self._text),
            metadata={"provider": PROVIDER_NAME},
        )

    async def aclose(self) -> None:
        self._closed = True
        if self._connection is None:
            return
        try:
            await self._connection.close()
        except Exception:  # noqa: BLE001 - closing must not raise
            logger.debug("ElevenLabs socket did not close cleanly.", exc_info=True)
        finally:
            self._connection = None


class ElevenLabsStreamingTextToSpeech:
    """Opens synthesis sockets. One per reply."""

    def __init__(
        self,
        connect=None,
        api_key: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._api_key = api_key if api_key is not None else resolved.elevenlabs_api_key
        self._voice = voice or resolved.tts_voice
        self._model = model or resolved.tts_model
        rate = sample_rate or resolved.audio_sample_rate
        if rate not in OUTPUT_FORMATS:
            raise VoiceError(
                f"ElevenLabs has no raw PCM output at {rate} Hz; available: "
                f"{', '.join(str(known) for known in OUTPUT_FORMATS)}."
            )
        self._output_format = OUTPUT_FORMATS[rate]
        self._format = AudioFormat(
            encoding=PCM_S16LE, sample_rate=rate, channels=1, container="raw"
        )
        self._connect = connect
        self._connect_timeout = resolved.stream_connect_timeout_seconds
        self._read_timeout = resolved.stream_read_timeout_seconds

    @property
    def format(self) -> AudioFormat:
        return self._format

    def stream(self, text: str, voice: str | None = None) -> ElevenLabsVoiceStream:
        chosen = voice or self._voice
        if not chosen:
            raise VoiceError("No ElevenLabs voice is configured.")
        if not self._api_key:
            raise VoiceUnavailable("No ElevenLabs API key is configured.")
        return ElevenLabsVoiceStream(
            text=text,
            url=connection_url(chosen, self._model, self._output_format),
            api_key=self._api_key,
            audio_format=self._format,
            connect=self._connect,
            connect_timeout=self._connect_timeout,
            read_timeout=self._read_timeout,
        )
