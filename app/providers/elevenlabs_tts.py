"""ElevenLabs implementation of `TextToSpeech`, over plain HTTP.

The only module that knows an ElevenLabs request or response. No vendor SDK:
the endpoint takes JSON and answers with audio bytes, so `httpx` is the whole
dependency and the client is injectable.

Audio is requested as headerless 16 kHz PCM and wrapped in a WAV container
here, so that everything downstream — the harness, the browser, milestone 6 —
receives audio that describes itself rather than audio plus a spoken promise
about its sample rate.

Nothing here has ever been run against the real service in this repository.
"""

import time

import httpx

from app.config import Settings, get_settings
from app.providers.speech import PCM_S16LE, AudioFormat, build_wav
from app.providers.tts import Speech, VoiceError, VoiceUnavailable

ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
PROVIDER_NAME = "elevenlabs"
# ElevenLabs names raw PCM output by its rate. 16 kHz is the milestone-5
# contract, so the reply needs no resampling before it reaches the browser.
OUTPUT_FORMATS = {16000: "pcm_16000", 22050: "pcm_22050", 24000: "pcm_24000"}


class ElevenLabsTextToSpeech:
    """Synthesises one complete reply through ElevenLabs' HTTP API."""

    def __init__(
        self,
        client: httpx.Client | None = None,
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
                f"ElevenLabs has no raw PCM output at {rate} Hz; "
                f"available: {', '.join(str(known) for known in OUTPUT_FORMATS)}."
            )
        self._output_format = OUTPUT_FORMATS[rate]
        self._format = AudioFormat(
            encoding=PCM_S16LE, sample_rate=rate, channels=1, container="wav"
        )
        self._timeout = resolved.speech_timeout_seconds
        self._client = client or httpx.Client(timeout=self._timeout)

    @property
    def format(self) -> AudioFormat:
        return self._format

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        chosen = voice or self._voice
        if not chosen:
            raise VoiceError("No ElevenLabs voice is configured.")
        if not self._api_key:
            raise VoiceUnavailable("No ElevenLabs API key is configured.")

        payload = {"text": text, "output_format": self._output_format}
        if self._model:
            payload["model_id"] = self._model

        started = time.perf_counter()
        try:
            response = self._client.post(
                ENDPOINT.format(voice_id=chosen),
                params={"output_format": self._output_format},
                headers={
                    "xi-api-key": self._api_key,
                    "Accept": "audio/pcm",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise VoiceUnavailable(f"ElevenLabs timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                raise VoiceUnavailable(f"ElevenLabs returned {status}.") from exc
            raise VoiceError(f"ElevenLabs rejected the request ({status}).") from exc
        except httpx.HTTPError as exc:
            raise VoiceUnavailable(
                f"ElevenLabs could not be reached: {exc}"
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        pcm = response.content
        if not pcm:
            raise VoiceError("ElevenLabs returned no audio.")
        if len(pcm) % 2:
            raise VoiceError(
                "ElevenLabs returned an odd number of bytes, which cannot be "
                "16-bit samples."
            )

        frames = len(pcm) // 2
        return Speech(
            audio=build_wav(pcm, self._format),
            format=self._format,
            duration_ms=round(frames * 1000 / self._format.sample_rate),
            voice=chosen,
            latency_ms=latency_ms,
            provider_name=PROVIDER_NAME,
            characters=len(text),
            metadata={"output_format": self._output_format},
        )
