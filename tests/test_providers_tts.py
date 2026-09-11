"""The text-to-speech boundary: the interface, offline, and ElevenLabs.

No API key, no network. ElevenLabs is exercised through a mock transport.
"""

import httpx
import pytest

from app.config import Settings
from app.providers.elevenlabs_tts import ElevenLabsTextToSpeech
from app.providers.offline_tts import DEFAULT_VOICE, OfflineTextToSpeech
from app.providers.speech import PCM_S16LE, read_wav
from app.providers.tts import Speech, TextToSpeech, VoiceError, VoiceUnavailable

PCM = b"\x01\x02" * 8000  # 8000 frames at 16 kHz = half a second


def _settings(**overrides) -> Settings:
    fields = {
        "elevenlabs_api_key": "test-key",
        "tts_voice": "voice-abc",
        "tts_model": "eleven_flash_v2_5",
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)


def _elevenlabs(handler=None, *, settings: Settings | None = None):
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return (handler or (lambda _: httpx.Response(200, content=PCM)))(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    return (
        ElevenLabsTextToSpeech(client=client, settings=settings or _settings()),
        requests,
    )


# --- the interface --------------------------------------------------------


def test_a_fake_satisfies_the_protocol() -> None:
    from app.providers.speech import AudioFormat

    class Fake:
        @property
        def format(self) -> AudioFormat:
            return AudioFormat(PCM_S16LE, 16000, 1)

        def synthesize(self, text: str, voice: str | None = None) -> Speech:
            return Speech(audio=b"", format=self.format)

    model: TextToSpeech = Fake()
    assert model.synthesize("hello").format.sample_rate == 16000


def test_the_offline_provider_satisfies_the_protocol() -> None:
    checked: TextToSpeech = OfflineTextToSpeech()
    assert checked.synthesize("hello").audio


def test_the_elevenlabs_provider_satisfies_the_protocol() -> None:
    provider, _ = _elevenlabs()
    checked: TextToSpeech = provider
    assert checked.synthesize("hello").audio


# --- the offline provider -------------------------------------------------


def test_the_offline_provider_returns_playable_wav() -> None:
    speech = OfflineTextToSpeech().synthesize("Thanks for calling.")

    info = read_wav(speech.audio)
    assert info.format.sample_rate == 16000
    assert info.format.channels == 1
    assert info.format.encoding == PCM_S16LE


def test_the_offline_provider_length_follows_the_text() -> None:
    short = OfflineTextToSpeech().synthesize("Yes.")
    long = OfflineTextToSpeech().synthesize("Yes, " * 40)

    assert long.duration_ms > short.duration_ms


def test_the_offline_provider_is_deterministic() -> None:
    assert (
        OfflineTextToSpeech().synthesize("Hello there").audio
        == OfflineTextToSpeech().synthesize("Hello there").audio
    )


def test_the_offline_provider_counts_characters() -> None:
    assert OfflineTextToSpeech().synthesize("Hello").characters == 5


def test_the_offline_provider_says_what_it_is() -> None:
    """A tone, not a voice, and it admits so."""
    speech = OfflineTextToSpeech().synthesize("Hello")

    assert speech.provider_name == "offline"
    assert speech.voice == DEFAULT_VOICE
    assert speech.metadata["offline"] is True


def test_the_offline_provider_declares_the_rate_it_was_built_with() -> None:
    assert OfflineTextToSpeech(sample_rate=24000).format.sample_rate == 24000


# --- ElevenLabs: request construction --------------------------------------


def test_the_voice_is_in_the_url() -> None:
    provider, requests = _elevenlabs()

    provider.synthesize("Hello there")

    assert requests[0].url.path.endswith("/text-to-speech/voice-abc")


def test_the_text_and_model_are_sent_as_json() -> None:
    import json

    provider, requests = _elevenlabs()

    provider.synthesize("Hello there")

    payload = json.loads(requests[0].content)
    assert payload["text"] == "Hello there"
    assert payload["model_id"] == "eleven_flash_v2_5"


def test_raw_pcm_is_requested_at_the_configured_rate() -> None:
    """Asking for PCM means no decoding, and no native dependency."""
    provider, requests = _elevenlabs()

    provider.synthesize("Hello")

    assert requests[0].url.params["output_format"] == "pcm_16000"


def test_the_key_is_sent() -> None:
    provider, requests = _elevenlabs()

    provider.synthesize("Hello")

    assert requests[0].headers["xi-api-key"] == "test-key"


def test_a_per_call_voice_overrides_the_configured_one() -> None:
    provider, requests = _elevenlabs()

    speech = provider.synthesize("Hello", voice="voice-xyz")

    assert requests[0].url.path.endswith("/text-to-speech/voice-xyz")
    assert speech.voice == "voice-xyz"


def test_no_voice_is_a_failure_before_any_request() -> None:
    provider, requests = _elevenlabs(settings=_settings(tts_voice=""))

    with pytest.raises(VoiceError, match="No ElevenLabs voice"):
        provider.synthesize("Hello")
    assert requests == []


def test_no_key_is_a_failure_before_any_request() -> None:
    provider, requests = _elevenlabs(settings=_settings(elevenlabs_api_key=""))

    with pytest.raises(VoiceUnavailable, match="No ElevenLabs API key"):
        provider.synthesize("Hello")
    assert requests == []


def test_a_rate_with_no_raw_pcm_output_is_refused_at_construction() -> None:
    """Better to fail building the provider than mid-call."""
    with pytest.raises(VoiceError, match="no raw PCM output"):
        ElevenLabsTextToSpeech(
            client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
            settings=_settings(audio_sample_rate=44100),
        )


# --- ElevenLabs: response parsing ------------------------------------------


def test_the_pcm_comes_back_wrapped_in_a_wav_container() -> None:
    """So the browser and milestone 6 get audio that describes itself."""
    provider, _ = _elevenlabs()

    speech = provider.synthesize("Hello")

    info = read_wav(speech.audio)
    assert info.pcm == PCM
    assert info.format.sample_rate == 16000


def test_the_duration_is_computed_from_the_samples() -> None:
    provider, _ = _elevenlabs()

    assert provider.synthesize("Hello").duration_ms == 500


def test_characters_and_latency_are_recorded() -> None:
    provider, _ = _elevenlabs()

    speech = provider.synthesize("Hello there")

    assert speech.characters == 11
    assert speech.latency_ms >= 0
    assert speech.provider_name == "elevenlabs"


# --- ElevenLabs: failures ---------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502], ids=["rate-limit", "500", "502"])
def test_rate_limits_and_server_errors_are_unavailable(status: int) -> None:
    provider, _ = _elevenlabs(lambda r: httpx.Response(status, content=b"nope"))

    with pytest.raises(VoiceUnavailable):
        provider.synthesize("Hello")


def test_a_bad_request_is_an_error_but_not_unavailable() -> None:
    provider, _ = _elevenlabs(lambda r: httpx.Response(422, content=b"bad voice"))

    with pytest.raises(VoiceError) as caught:
        provider.synthesize("Hello")
    assert not isinstance(caught.value, VoiceUnavailable)


def test_a_timeout_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    provider, _ = _elevenlabs(handler)

    with pytest.raises(VoiceUnavailable, match="timed out"):
        provider.synthesize("Hello")


def test_a_connection_failure_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    provider, _ = _elevenlabs(handler)

    with pytest.raises(VoiceUnavailable, match="could not be reached"):
        provider.synthesize("Hello")


def test_an_empty_body_is_an_error_not_silent_audio() -> None:
    provider, _ = _elevenlabs(lambda r: httpx.Response(200, content=b""))

    with pytest.raises(VoiceError, match="no audio"):
        provider.synthesize("Hello")


def test_an_odd_number_of_bytes_cannot_be_sixteen_bit_samples() -> None:
    provider, _ = _elevenlabs(lambda r: httpx.Response(200, content=b"\x01\x02\x03"))

    with pytest.raises(VoiceError, match="odd number of bytes"):
        provider.synthesize("Hello")
