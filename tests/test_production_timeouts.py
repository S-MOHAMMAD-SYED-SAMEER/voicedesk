"""Every outbound boundary has a clock on it, and an honest one.

Before this milestone the application waited on four things without any
bound: the model SDK's own default (ten minutes), a streaming socket that
connects and then says nothing (for ever), and the operating system's TCP
timeout on a database that is not there (minutes). A caller is on the
telephone for all three.

What a timeout here does and does not do is stated once, in
`app/providers/streaming.py` and again in the README: it ends the **wait**.
The model request runs on a worker thread, and when the wait ends that thread
is abandoned, not killed — Python offers no way to kill it. These tests
assert the wait ends. None of them claims the work stopped.
"""

import inspect

import anyio
import pytest

from app.config import Settings
from app.providers.streaming import messages


class _Silent:
    """A connection that opens and then never says anything."""

    def __init__(self, *, before: list | None = None) -> None:
        self.before = list(before or [])

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.before:
            return self.before.pop(0)
        await anyio.sleep(60)
        raise AssertionError("the sleep should have been cut short")


class _Talkative:
    def __init__(self, *items) -> None:
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        await anyio.sleep(0.01)
        return self._items.pop(0)


# --- the idle clock on a streaming socket ----------------------------------


@pytest.mark.anyio
async def test_every_message_is_passed_through() -> None:
    assert [item async for item in messages(_Talkative("a", "b", "c"), 5.0)] == [
        "a",
        "b",
        "c",
    ]


@pytest.mark.anyio
async def test_the_end_of_the_stream_ends_the_iteration() -> None:
    assert [item async for item in messages(_Talkative(), 5.0)] == []


@pytest.mark.anyio
async def test_a_connection_that_goes_quiet_raises() -> None:
    """The failure mode neither vendor protocol has an answer for."""
    with pytest.raises(TimeoutError):
        async for _ in messages(_Silent(), 0.05):
            pass


@pytest.mark.anyio
async def test_what_arrived_before_the_silence_still_arrived() -> None:
    received = []

    with pytest.raises(TimeoutError):
        async for item in messages(_Silent(before=["first", "second"]), 0.05):
            received.append(item)

    assert received == ["first", "second"]


@pytest.mark.anyio
async def test_the_clock_is_between_messages_not_around_the_whole_stream() -> None:
    """A synthesiser sending audio steadily is working, however long the
    reply is. Ten messages at 10 ms each must not trip a 50 ms idle clock."""
    connection = _Talkative(*range(10))

    assert len([item async for item in messages(connection, 0.05)]) == 10


# --- the streaming adapters use it -----------------------------------------


@pytest.mark.anyio
async def test_the_recogniser_reports_a_silent_socket_as_unavailable() -> None:
    """It becomes the adapter's own error, so a quiet provider fails exactly
    like a disconnected one and the existing handler catches it."""
    from app.providers.deepgram_stream_stt import DeepgramSpeechStream
    from app.providers.speech import PCM_S16LE, AudioFormat
    from app.providers.streaming_stt import SpeechUnavailable

    stream = DeepgramSpeechStream(
        _Silent(), AudioFormat(PCM_S16LE, 8000, 1, "raw"), 0.05
    )

    with pytest.raises(SpeechUnavailable):
        async for _ in stream.events():
            pass


@pytest.mark.anyio
async def test_a_recogniser_connection_that_never_opens_is_unavailable() -> None:
    from app.providers.deepgram_stream_stt import DeepgramStreamingSpeechToText
    from app.providers.speech import PCM_S16LE, AudioFormat
    from app.providers.streaming_stt import SpeechUnavailable

    async def never(url, **kwargs):
        await anyio.sleep(60)

    provider = DeepgramStreamingSpeechToText(
        connect=never,
        settings=Settings(
            _env_file=None,
            deepgram_api_key="test-key",
            stream_connect_timeout_seconds=0.05,
        ),
    )
    stream = provider.stream(AudioFormat(PCM_S16LE, 8000, 1, "raw"))

    with pytest.raises(SpeechUnavailable, match="could not be reached"):
        await stream.send(b"\x00\x00" * 160)


@pytest.mark.anyio
async def test_a_synthesiser_connection_that_never_opens_is_unavailable() -> None:
    from app.providers.elevenlabs_stream_tts import ElevenLabsStreamingTextToSpeech
    from app.providers.streaming_tts import VoiceUnavailable

    async def never(url, **kwargs):
        await anyio.sleep(60)

    provider = ElevenLabsStreamingTextToSpeech(
        connect=never,
        settings=Settings(
            _env_file=None,
            elevenlabs_api_key="test-key",
            tts_voice="voice-abc",
            stream_connect_timeout_seconds=0.05,
        ),
    )

    with pytest.raises(VoiceUnavailable, match="could not be reached"):
        async for _ in provider.stream("Hello.").chunks():
            pass


@pytest.mark.anyio
async def test_a_synthesiser_that_goes_quiet_is_unavailable() -> None:
    from app.providers.elevenlabs_stream_tts import ElevenLabsVoiceStream
    from app.providers.speech import PCM_S16LE, AudioFormat
    from app.providers.streaming_tts import VoiceUnavailable

    silent = _Silent()

    async def connect(url, **kwargs):
        silent.sent = []
        silent.send = _record(silent)
        silent.close = _closed(silent)
        return silent

    stream = ElevenLabsVoiceStream(
        text="Hello.",
        url="wss://example.invalid/stream",
        api_key="test-key",
        audio_format=AudioFormat(PCM_S16LE, 8000, 1, "raw"),
        connect=connect,
        read_timeout=0.05,
    )

    with pytest.raises(VoiceUnavailable):
        async for _ in stream.chunks():
            pass


def _record(connection):
    async def send(message) -> None:
        connection.sent.append(message)

    return send


def _closed(connection):
    async def close() -> None:
        connection.was_closed = True

    return close


# --- the timeouts that are configuration, not behaviour --------------------


def test_the_model_client_is_built_with_a_timeout() -> None:
    """The SDK's own default is minutes, which is not a telephone call."""
    from app.providers.anthropic_llm import AnthropicLanguageModel

    captured = {}

    class _Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import app.providers.anthropic_llm as module

    original = module.anthropic.Anthropic
    module.anthropic.Anthropic = _Client
    try:
        AnthropicLanguageModel(
            settings=Settings(
                _env_file=None,
                anthropic_api_key="sk-test",
                dialogue_timeout_seconds=7.5,
            )
        )
    finally:
        module.anthropic.Anthropic = original

    assert captured["timeout"] == 7.5


def test_the_database_engine_is_built_with_a_connect_timeout(
    settings_env, monkeypatch
) -> None:
    from app.db import session as module

    captured = {}

    def _engine(url, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "create_engine", _engine)
    monkeypatch.setattr(module, "get_settings", lambda: Settings(_env_file=None))
    module.get_engine.cache_clear()
    try:
        module.get_engine()
    finally:
        module.get_engine.cache_clear()

    assert captured["connect_args"]["connect_timeout"] == 10


@pytest.mark.parametrize(
    "field",
    [
        "dialogue_timeout_seconds",
        "stream_connect_timeout_seconds",
        "stream_read_timeout_seconds",
        "database_connect_timeout_seconds",
        "shutdown_grace_seconds",
    ],
)
def test_no_timeout_can_be_configured_to_zero(field: str) -> None:
    """Zero is not "no timeout"; it is a boundary that fails immediately."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: 0})


# --- the limitation this milestone does not pretend away -------------------


def test_the_abandoned_thread_is_documented_rather_than_claimed() -> None:
    """`to_thread.run_sync` abandons; it cannot kill. Saying so is the fix
    that is available, and the code says so where somebody will read it."""
    from app import runtime

    text = " ".join(
        source
        for source in (
            inspect.getsource(runtime),
            inspect.getsource(__import__("app.config", fromlist=["x"])),
        )
    ).lower()

    assert "cannot be cancelled" in text or "cannot kill" in text
