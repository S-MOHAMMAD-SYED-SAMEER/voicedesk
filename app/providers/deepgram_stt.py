"""Deepgram implementation of `SpeechToText`, over plain HTTP.

This is the only module that knows what a Deepgram request or response looks
like. There is no vendor SDK: the pre-recorded endpoint takes the audio as the
request body and answers with JSON, so `httpx` is the whole dependency, and
the client is injectable — which is how the tests exercise the request shape
and every failure path without a key or a network.

Nothing here has ever been run against the real service in this repository.
The request shape is verified; the transcription accuracy is not claimed.
"""

import time
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.providers.speech import Audio
from app.providers.stt import (
    SpeechError,
    SpeechUnavailable,
    Transcript,
    UnsupportedAudio,
)

ENDPOINT = "https://api.deepgram.com/v1/listen"
PROVIDER_NAME = "deepgram"
# What the milestone-5 audio contract produces, and the only thing this
# adapter offers to send. The container is in the bytes, so Deepgram reads the
# rate and width from the WAV header itself.
CONTENT_TYPE = "audio/wav"


class DeepgramSpeechToText:
    """Transcribes one complete recording through Deepgram's HTTP API."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        self._api_key = api_key if api_key is not None else resolved.deepgram_api_key
        self._model = model or resolved.stt_model
        self._timeout = resolved.speech_timeout_seconds
        self._client = client or httpx.Client(timeout=self._timeout)

    def transcribe(self, audio: Audio) -> Transcript:
        if audio.format.container != "wav":
            raise UnsupportedAudio(
                f"This adapter sends WAV; it was handed {audio.format.container!r}."
            )
        if not self._api_key:
            raise SpeechUnavailable("No Deepgram API key is configured.")

        started = time.perf_counter()
        try:
            response = self._client.post(
                ENDPOINT,
                params={"model": self._model, "smart_format": "true"},
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": CONTENT_TYPE,
                },
                content=audio.data,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise SpeechUnavailable(f"Deepgram timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                raise SpeechUnavailable(f"Deepgram returned {status}.") from exc
            raise SpeechError(f"Deepgram rejected the request ({status}).") from exc
        except httpx.HTTPError as exc:
            raise SpeechUnavailable(f"Deepgram could not be reached: {exc}") from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        return _read(response, latency_ms)


def _read(response: httpx.Response, latency_ms: int) -> Transcript:
    """Deepgram's JSON, translated into a `Transcript`.

    A response that cannot be read is an error, never an empty transcript —
    silence and a broken answer must not look the same to the caller.
    """
    try:
        body: dict[str, Any] = response.json()
    except ValueError as exc:
        raise SpeechError("Deepgram returned something that is not JSON.") from exc

    try:
        channel = body["results"]["channels"][0]
        alternative = channel["alternatives"][0]
        text = alternative["transcript"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SpeechError(
            f"Deepgram's response had no transcript in it: {exc}"
        ) from exc

    if not isinstance(text, str):
        raise SpeechError("Deepgram returned a transcript that is not text.")

    duration_seconds = body.get("metadata", {}).get("duration")
    return Transcript(
        text=text.strip(),
        confidence=alternative.get("confidence"),
        language=channel.get("detected_language"),
        audio_ms=(
            round(duration_seconds * 1000)
            if isinstance(duration_seconds, int | float)
            else None
        ),
        latency_ms=latency_ms,
        provider_name=PROVIDER_NAME,
        metadata={"request_id": body.get("metadata", {}).get("request_id")},
    )
