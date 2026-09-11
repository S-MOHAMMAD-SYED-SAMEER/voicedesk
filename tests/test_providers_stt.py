"""The speech-to-text boundary: the interface, offline, and Deepgram.

No API key, no network. Deepgram is exercised through a mock transport, so the
request shape is verified without a request ever leaving the process.
"""

import httpx
import pytest

from app.config import Settings
from app.providers.deepgram_stt import ENDPOINT, DeepgramSpeechToText
from app.providers.offline_stt import DEFAULT_TRANSCRIPT, OfflineSpeechToText
from app.providers.speech import PCM_S16LE, Audio, AudioFormat
from app.providers.stt import (
    SpeechError,
    SpeechToText,
    SpeechUnavailable,
    Transcript,
    UnsupportedAudio,
)

from .conftest import wav_bytes

FORMAT = AudioFormat(PCM_S16LE, 16000, 1, "wav")


def _audio(duration_ms: int = 500) -> Audio:
    return Audio(data=wav_bytes(duration_ms), format=FORMAT, duration_ms=duration_ms)


def _body(
    transcript: str = "I'd like to book a haircut",
    confidence: float | None = 0.97,
    duration: float | None = 1.5,
    language: str | None = "en",
) -> dict:
    alternative: dict = {"transcript": transcript}
    if confidence is not None:
        alternative["confidence"] = confidence
    channel: dict = {"alternatives": [alternative]}
    if language is not None:
        channel["detected_language"] = language
    body: dict = {"results": {"channels": [channel]}, "metadata": {"request_id": "r1"}}
    if duration is not None:
        body["metadata"]["duration"] = duration
    return body


def _deepgram(handler, *, settings: Settings | None = None):
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    resolved = settings or Settings(_env_file=None, deepgram_api_key="test-key")
    return DeepgramSpeechToText(client=client, settings=resolved), requests


def _answers(body: dict | None = None, status: int = 200, content: bytes | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if content is not None:
            return httpx.Response(status, content=content)
        return httpx.Response(status, json=body if body is not None else _body())

    return handler


# --- the interface --------------------------------------------------------


def test_a_fake_satisfies_the_protocol() -> None:
    class Fake:
        def transcribe(self, audio: Audio) -> Transcript:
            return Transcript(text="hello")

    model: SpeechToText = Fake()
    assert model.transcribe(_audio()).text == "hello"


def test_the_offline_provider_satisfies_the_protocol() -> None:
    checked: SpeechToText = OfflineSpeechToText()
    assert checked.transcribe(_audio()).text == DEFAULT_TRANSCRIPT


def test_the_deepgram_provider_satisfies_the_protocol() -> None:
    provider, _ = _deepgram(_answers())
    checked: SpeechToText = provider
    assert checked.transcribe(_audio()).text


def test_an_empty_transcript_is_a_transcript_not_an_error() -> None:
    """Silence happened. A provider outage did not."""
    assert Transcript(text="").text == ""


# --- the offline provider -------------------------------------------------


def test_the_offline_provider_reads_the_duration_from_the_audio() -> None:
    assert OfflineSpeechToText().transcribe(_audio(1250)).audio_ms == 1250


def test_the_offline_provider_can_be_scripted() -> None:
    provider = OfflineSpeechToText(transcripts=["first", "second"])

    assert [provider.transcribe(_audio()).text for _ in range(3)] == [
        "first",
        "second",
        "second",
    ]


def test_the_offline_provider_refuses_audio_it_cannot_read() -> None:
    with pytest.raises(UnsupportedAudio):
        OfflineSpeechToText().transcribe(
            Audio(data=b"not a wav", format=FORMAT)
        )


def test_the_offline_provider_says_what_it_is() -> None:
    """It is a harness, not speech recognition, and it admits so."""
    transcript = OfflineSpeechToText().transcribe(_audio())

    assert transcript.provider_name == "offline"
    assert transcript.metadata["offline"] is True


# --- Deepgram: request construction ---------------------------------------


def test_the_audio_is_sent_as_the_request_body() -> None:
    provider, requests = _deepgram(_answers())
    audio = _audio()

    provider.transcribe(audio)

    assert requests[0].content == audio.data
    assert str(requests[0].url).startswith(ENDPOINT)


def test_the_key_and_content_type_are_sent() -> None:
    provider, requests = _deepgram(_answers())

    provider.transcribe(_audio())

    assert requests[0].headers["authorization"] == "Token test-key"
    assert requests[0].headers["content-type"] == "audio/wav"


def test_the_model_comes_from_settings() -> None:
    provider, requests = _deepgram(
        _answers(),
        settings=Settings(
            _env_file=None, deepgram_api_key="k", stt_model="nova-3-medical"
        ),
    )

    provider.transcribe(_audio())

    assert requests[0].url.params["model"] == "nova-3-medical"


def test_no_key_is_a_failure_before_any_request() -> None:
    provider, requests = _deepgram(
        _answers(), settings=Settings(_env_file=None, deepgram_api_key="")
    )

    with pytest.raises(SpeechUnavailable, match="No Deepgram API key"):
        provider.transcribe(_audio())
    assert requests == []


def test_audio_that_is_not_a_wav_container_is_refused_before_any_request() -> None:
    provider, requests = _deepgram(_answers())
    raw = Audio(data=b"\x00\x00", format=AudioFormat(PCM_S16LE, 16000, 1, "raw"))

    with pytest.raises(UnsupportedAudio):
        provider.transcribe(raw)
    assert requests == []


# --- Deepgram: response parsing -------------------------------------------


def test_the_transcript_is_read_back() -> None:
    provider, _ = _deepgram(_answers())

    transcript = provider.transcribe(_audio())

    assert transcript.text == "I'd like to book a haircut"
    assert transcript.provider_name == "deepgram"


def test_confidence_language_and_duration_are_mapped() -> None:
    provider, _ = _deepgram(_answers(_body(confidence=0.81, duration=2.0)))

    transcript = provider.transcribe(_audio())

    assert transcript.confidence == 0.81
    assert transcript.language == "en"
    assert transcript.audio_ms == 2000


def test_missing_confidence_is_none_rather_than_a_guess() -> None:
    provider, _ = _deepgram(_answers(_body(confidence=None, duration=None)))

    transcript = provider.transcribe(_audio())

    assert transcript.confidence is None
    assert transcript.audio_ms is None


def test_latency_is_measured() -> None:
    provider, _ = _deepgram(_answers())

    assert provider.transcribe(_audio()).latency_ms >= 0


def test_an_empty_transcript_comes_back_empty() -> None:
    provider, _ = _deepgram(_answers(_body(transcript="")))

    transcript = provider.transcribe(_audio())

    assert transcript.text == ""


def test_surrounding_whitespace_is_trimmed() -> None:
    provider, _ = _deepgram(_answers(_body(transcript="  hello  ")))

    assert provider.transcribe(_audio()).text == "hello"


# --- Deepgram: failures ----------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 503], ids=["rate-limit", "500", "503"])
def test_rate_limits_and_server_errors_are_unavailable(status: int) -> None:
    provider, _ = _deepgram(_answers(status=status, content=b"nope"))

    with pytest.raises(SpeechUnavailable):
        provider.transcribe(_audio())


def test_a_bad_request_is_an_error_but_not_unavailable() -> None:
    """Retrying a 400 would just fail again; it is our bug, not their outage."""
    provider, _ = _deepgram(_answers(status=400, content=b"bad"))

    with pytest.raises(SpeechError) as caught:
        provider.transcribe(_audio())
    assert not isinstance(caught.value, SpeechUnavailable)


def test_a_timeout_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider, _ = _deepgram(handler)

    with pytest.raises(SpeechUnavailable, match="timed out"):
        provider.transcribe(_audio())


def test_a_connection_failure_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    provider, _ = _deepgram(handler)

    with pytest.raises(SpeechUnavailable, match="could not be reached"):
        provider.transcribe(_audio())


def test_a_response_that_is_not_json_is_an_error() -> None:
    provider, _ = _deepgram(_answers(content=b"<html>down</html>"))

    with pytest.raises(SpeechError, match="not JSON"):
        provider.transcribe(_audio())


@pytest.mark.parametrize(
    "body",
    [{}, {"results": {}}, {"results": {"channels": []}},
     {"results": {"channels": [{"alternatives": []}]}}],
    ids=["empty", "no-channels", "empty-channels", "no-alternatives"],
)
def test_a_response_with_no_transcript_is_an_error_not_silence(body: dict) -> None:
    """A broken answer must never be mistaken for the caller saying nothing."""
    provider, _ = _deepgram(_answers(body))

    with pytest.raises(SpeechError, match="no transcript"):
        provider.transcribe(_audio())


def test_a_transcript_that_is_not_text_is_an_error() -> None:
    provider, _ = _deepgram(_answers({"results": {"channels": [
        {"alternatives": [{"transcript": 42}]}
    ]}}))

    with pytest.raises(SpeechError, match="not text"):
        provider.transcribe(_audio())
