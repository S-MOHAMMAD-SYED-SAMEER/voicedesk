"""Building the configured speech providers, by name.

One place that knows which setting selects which adapter, so the harness (and
milestone 6's telephony after it) asks for "the configured transcriber" rather
than importing a vendor module and deciding for itself.

Imports happen inside the functions on purpose: selecting the offline provider
must not load an adapter that is not going to be used, and importing this
module must never drag in a vendor's request shapes.
"""

from app.config import Settings, get_settings
from app.providers.llm import LanguageModel
from app.providers.streaming_stt import StreamingSpeechToText
from app.providers.streaming_tts import StreamingTextToSpeech
from app.providers.stt import SpeechToText
from app.providers.tts import TextToSpeech


def build_stt(settings: Settings | None = None) -> SpeechToText:
    """The transcriber named by `stt_provider`."""
    resolved = settings or get_settings()
    if resolved.stt_provider == "deepgram":
        from app.providers.deepgram_stt import DeepgramSpeechToText

        return DeepgramSpeechToText(settings=resolved)

    from app.providers.offline_stt import OfflineSpeechToText

    return OfflineSpeechToText()


def build_tts(settings: Settings | None = None) -> TextToSpeech:
    """The synthesiser named by `tts_provider`."""
    resolved = settings or get_settings()
    if resolved.tts_provider == "elevenlabs":
        from app.providers.elevenlabs_tts import ElevenLabsTextToSpeech

        return ElevenLabsTextToSpeech(settings=resolved)

    from app.providers.offline_tts import OfflineTextToSpeech

    return OfflineTextToSpeech(sample_rate=resolved.audio_sample_rate)


def build_model(settings: Settings | None = None) -> LanguageModel:
    """The configured language model.

    Here for the same reason as the two above: one place decides which
    implementation a caller gets, so nothing else has to import a vendor
    module to make that choice.
    """
    from app.providers.anthropic_llm import AnthropicLanguageModel

    return AnthropicLanguageModel(settings=settings or get_settings())


def build_streaming_stt(settings: Settings | None = None) -> StreamingSpeechToText:
    """The streaming transcriber named by `stt_streaming_provider`."""
    resolved = settings or get_settings()
    if resolved.stt_streaming_provider == "deepgram":
        from app.providers.deepgram_stream_stt import DeepgramStreamingSpeechToText

        return DeepgramStreamingSpeechToText(settings=resolved)

    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    return OfflineStreamingSpeechToText()


def build_streaming_tts(settings: Settings | None = None) -> StreamingTextToSpeech:
    """The streaming synthesiser named by `tts_streaming_provider`."""
    resolved = settings or get_settings()
    if resolved.tts_streaming_provider == "elevenlabs":
        from app.providers.elevenlabs_stream_tts import ElevenLabsStreamingTextToSpeech

        return ElevenLabsStreamingTextToSpeech(settings=resolved)

    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    return OfflineStreamingTextToSpeech(sample_rate=resolved.audio_sample_rate)
