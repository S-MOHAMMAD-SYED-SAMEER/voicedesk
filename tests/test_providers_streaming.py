"""The streaming boundaries: the interfaces, offline, and the two adapters.

Nothing here needs a key or a network. The vendor adapters are driven through
a fake connection, so their frame shapes are exercised without a request
leaving the process.
"""

import base64
import json

import pytest

from app.config import Settings
from app.providers.speech import PCM_S16LE, AudioFormat
from app.providers.streaming_stt import (
    FinalTranscript,
    PartialTranscript,
    SpeechError,
    SpeechUnavailable,
    StreamingSpeechToText,
    UnsupportedAudio,
)
from app.providers.streaming_tts import (
    SpeechChunk,
    StreamingTextToSpeech,
    VoiceError,
    VoiceUnavailable,
)

FORMAT = AudioFormat(PCM_S16LE, 8000, 1, "raw")
PCM = b"\x01\x02" * 800


async def _drain(stream, *, finish: bool = True) -> list:
    """Collect every event, telling the stream the utterance is over."""
    import anyio

    events: list = []

    async def pump() -> None:
        async for event in stream.events():
            events.append(event)

    async with anyio.create_task_group() as group:
        group.start_soon(pump)
        await anyio.sleep(0)
        if finish:
            await stream.finish()
    return events


class FakeConnection:
    """Stands in for a WebSocket: scripted inbound, recorded outbound."""

    def __init__(self, *messages, raises: Exception | None = None) -> None:
        self._messages = list(messages)
        self._raises = raises
        self.sent: list = []
        self.closed = False

    async def send(self, message) -> None:
        if self._raises is not None:
            raise self._raises
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        message = self._messages.pop(0)
        if isinstance(message, Exception):
            raise message
        return message


# --- the interfaces -------------------------------------------------------


def test_the_offline_transcriber_satisfies_the_protocol() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    checked: StreamingSpeechToText = OfflineStreamingSpeechToText()
    assert checked.stream(FORMAT) is not None


def test_the_offline_synthesiser_satisfies_the_protocol() -> None:
    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    checked: StreamingTextToSpeech = OfflineStreamingTextToSpeech()
    assert checked.format.sample_rate == 16000


def test_a_chunk_knows_how_long_it_plays_for() -> None:
    chunk = SpeechChunk(audio=b"\x00\x00" * 8000, format=FORMAT)

    assert chunk.duration_ms == 1000


def test_an_empty_chunk_plays_for_no_time() -> None:
    assert SpeechChunk(audio=b"", format=FORMAT).duration_ms == 0


# --- the offline transcriber ----------------------------------------------


@pytest.mark.anyio
async def test_partials_arrive_before_the_final() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    stream = OfflineStreamingSpeechToText().stream(FORMAT)
    await stream.send(PCM)

    events = await _drain(stream)

    assert [type(event).__name__ for event in events] == [
        "PartialTranscript",
        "PartialTranscript",
        "FinalTranscript",
    ]


@pytest.mark.anyio
async def test_exactly_one_final_arrives() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    stream = OfflineStreamingSpeechToText().stream(FORMAT)
    await stream.send(PCM)

    events = await _drain(stream)

    assert len([e for e in events if isinstance(e, FinalTranscript)]) == 1


@pytest.mark.anyio
async def test_the_final_carries_confidence_and_duration() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    stream = OfflineStreamingSpeechToText(confidence=0.81).stream(FORMAT)
    await stream.send(PCM)

    final = [e for e in await _drain(stream) if isinstance(e, FinalTranscript)][0]

    assert final.confidence == 0.81
    assert final.audio_ms == 100
    assert final.provider_name == "offline"


@pytest.mark.anyio
async def test_the_transcript_can_be_scripted() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    stream = OfflineStreamingSpeechToText(
        partials=("one",), final="cancel my appointment"
    ).stream(FORMAT)
    await stream.send(PCM)

    events = await _drain(stream)

    assert [e.text for e in events] == ["one", "cancel my appointment"]


@pytest.mark.anyio
async def test_a_scripted_failure_is_raised() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    stream = OfflineStreamingSpeechToText(
        raises=SpeechUnavailable("the provider fell over")
    ).stream(FORMAT)

    with pytest.raises(SpeechUnavailable):
        # Raised before anything is yielded, so no task group is needed to
        # see it — which is also how the session encounters it.
        [event async for event in stream.events()]


@pytest.mark.anyio
async def test_closing_stops_the_events() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    stream = OfflineStreamingSpeechToText().stream(FORMAT)
    await stream.send(PCM)
    await stream.aclose()

    assert await _drain(stream, finish=False) == []


@pytest.mark.anyio
async def test_sending_to_a_closed_stream_is_an_error() -> None:
    from app.providers.offline_streaming import OfflineStreamingSpeechToText

    stream = OfflineStreamingSpeechToText().stream(FORMAT)
    await stream.aclose()

    with pytest.raises(SpeechError):
        await stream.send(PCM)


# --- the offline synthesiser ----------------------------------------------


@pytest.mark.anyio
async def test_audio_arrives_in_chunks_ending_with_a_final_one() -> None:
    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    chunks = [c async for c in OfflineStreamingTextToSpeech().stream("Hello there").chunks()]

    assert len(chunks) > 1
    assert chunks[-1].is_final
    assert not any(chunk.is_final for chunk in chunks[:-1])


@pytest.mark.anyio
async def test_chunks_carry_headerless_pcm() -> None:
    """A WAV header per chunk would be one the caller cannot chain."""
    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    chunk = [c async for c in OfflineStreamingTextToSpeech().stream("Hi").chunks()][0]

    assert chunk.format.container == "raw"
    assert not chunk.audio.startswith(b"RIFF")


@pytest.mark.anyio
async def test_a_longer_reply_is_longer_audio() -> None:
    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    async def spoken_ms(text: str) -> float:
        return sum(
            [c.duration_ms async for c in OfflineStreamingTextToSpeech().stream(text).chunks()]
        )

    assert await spoken_ms("Yes, " * 40) > await spoken_ms("Yes.")


@pytest.mark.anyio
async def test_closing_stops_the_chunks() -> None:
    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    stream = OfflineStreamingTextToSpeech().stream("Hello there, how can I help?")
    seen = 0
    async for _ in stream.chunks():
        seen += 1
        await stream.aclose()

    assert seen == 1


@pytest.mark.anyio
async def test_a_scripted_synthesis_failure_is_raised() -> None:
    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    stream = OfflineStreamingTextToSpeech(raises=VoiceUnavailable("down")).stream("Hi")

    with pytest.raises(VoiceUnavailable):
        [c async for c in stream.chunks()]


@pytest.mark.anyio
async def test_synthesis_is_deterministic() -> None:
    from app.providers.offline_streaming import OfflineStreamingTextToSpeech

    async def audio() -> bytes:
        return b"".join(
            [c.audio async for c in OfflineStreamingTextToSpeech().stream("Hello").chunks()]
        )

    assert await audio() == await audio()


# --- the Deepgram adapter -------------------------------------------------


def _deepgram(*messages, raises=None, **overrides):
    from app.providers.deepgram_stream_stt import DeepgramStreamingSpeechToText

    connection = FakeConnection(*messages, raises=raises)

    async def connect(url, **kwargs):
        connect.url, connect.kwargs = url, kwargs
        return connection

    fields = {"deepgram_api_key": "test-key", "stt_model": "nova-3"}
    fields.update(overrides)
    provider = DeepgramStreamingSpeechToText(
        connect=connect, settings=Settings(_env_file=None, **fields)
    )
    return provider, connection, connect


def _result(text: str, *, speech_final: bool = False, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "type": "Results",
            "is_final": speech_final,
            "speech_final": speech_final,
            "duration": 1.5,
            "start": 0.5,
            "channel": {
                "alternatives": [{"transcript": text, "confidence": confidence}]
            },
        }
    )


@pytest.mark.anyio
async def test_the_url_describes_the_audio_being_sent() -> None:
    provider, _, connect = _deepgram(_result("hi", speech_final=True))
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    assert "encoding=linear16" in connect.url
    assert "sample_rate=8000" in connect.url
    assert "model=nova-3" in connect.url
    assert connect.kwargs["additional_headers"]["Authorization"] == "Token test-key"


@pytest.mark.anyio
async def test_audio_is_sent_as_binary_frames() -> None:
    provider, connection, _ = _deepgram(_result("hi", speech_final=True))
    stream = provider.stream(FORMAT)

    await stream.send(PCM)

    assert connection.sent == [PCM]


@pytest.mark.anyio
async def test_finishing_asks_the_service_to_close() -> None:
    provider, connection, _ = _deepgram(_result("hi", speech_final=True))
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    await stream.finish()

    assert json.loads(connection.sent[-1]) == {"type": "CloseStream"}


@pytest.mark.anyio
async def test_interim_results_are_partials_and_the_last_is_final() -> None:
    provider, _, _ = _deepgram(
        _result("I'd"), _result("I'd like"), _result("I'd like a haircut", speech_final=True)
    )
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    events = [event async for event in stream.events()]

    assert [type(event).__name__ for event in events] == [
        "PartialTranscript",
        "PartialTranscript",
        "FinalTranscript",
    ]
    assert events[-1].text == "I'd like a haircut"
    assert events[-1].confidence == 0.9
    assert events[-1].audio_ms == 1500


@pytest.mark.anyio
async def test_bookkeeping_messages_are_skipped() -> None:
    provider, _, _ = _deepgram(
        json.dumps({"type": "Metadata"}),
        json.dumps({"type": "SpeechStarted"}),
        _result("hello", speech_final=True),
    )
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    events = [event async for event in stream.events()]

    assert len(events) == 1


@pytest.mark.anyio
async def test_an_empty_interim_result_is_not_a_partial() -> None:
    provider, _, _ = _deepgram(_result(""), _result("hello", speech_final=True))
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    assert len([e async for e in stream.events()]) == 1


@pytest.mark.anyio
async def test_a_reported_error_is_raised() -> None:
    provider, _, _ = _deepgram(json.dumps({"type": "Error", "description": "nope"}))
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    with pytest.raises(SpeechError, match="nope"):
        [event async for event in stream.events()]


@pytest.mark.anyio
async def test_a_response_that_is_not_json_is_an_error() -> None:
    provider, _, _ = _deepgram("<html>down</html>")
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    with pytest.raises(SpeechError):
        [event async for event in stream.events()]


@pytest.mark.anyio
async def test_a_disconnect_is_unavailable_and_is_not_reconnected() -> None:
    """A reconnect loses the audio the service had; silence would be a lie."""
    provider, _, _ = _deepgram(ConnectionResetError("gone"))
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    with pytest.raises(SpeechUnavailable):
        [event async for event in stream.events()]


@pytest.mark.anyio
async def test_a_send_failure_is_unavailable() -> None:
    provider, _, _ = _deepgram(raises=ConnectionResetError("gone"))
    stream = provider.stream(FORMAT)

    with pytest.raises(SpeechUnavailable):
        await stream.send(PCM)


def test_no_key_is_refused_before_anything_opens() -> None:
    provider, _, _ = _deepgram(deepgram_api_key="")

    with pytest.raises(SpeechUnavailable, match="No Deepgram API key"):
        provider.stream(FORMAT)


def test_audio_this_adapter_cannot_send_is_refused() -> None:
    provider, _, _ = _deepgram()

    with pytest.raises(UnsupportedAudio):
        provider.stream(AudioFormat("mulaw", 8000, 1, "raw"))


@pytest.mark.anyio
async def test_closing_closes_the_socket() -> None:
    provider, connection, _ = _deepgram(_result("hi", speech_final=True))
    stream = provider.stream(FORMAT)
    await stream.send(PCM)

    await stream.aclose()

    assert connection.closed


# --- the ElevenLabs adapter -----------------------------------------------


def _elevenlabs(*messages, **overrides):
    from app.providers.elevenlabs_stream_tts import ElevenLabsStreamingTextToSpeech

    connection = FakeConnection(*messages)

    async def connect(url, **kwargs):
        connect.url = url
        return connection

    fields = {
        "elevenlabs_api_key": "test-key",
        "tts_voice": "voice-abc",
        "tts_model": "eleven_flash_v2_5",
    }
    fields.update(overrides)
    provider = ElevenLabsStreamingTextToSpeech(
        connect=connect, settings=Settings(_env_file=None, **fields)
    )
    return provider, connection, connect


def _audio_message(pcm: bytes, *, final: bool = False) -> str:
    return json.dumps(
        {"audio": base64.b64encode(pcm).decode(), "isFinal": final}
    )


@pytest.mark.anyio
async def test_the_voice_and_format_are_in_the_url() -> None:
    provider, _, connect = _elevenlabs(_audio_message(b"\x01\x02", final=True))

    [c async for c in provider.stream("Hello").chunks()]

    assert "/text-to-speech/voice-abc/stream-input" in connect.url
    assert "output_format=pcm_16000" in connect.url
    assert "model_id=eleven_flash_v2_5" in connect.url


@pytest.mark.anyio
async def test_the_whole_reply_is_sent_then_closed() -> None:
    """The reply is already complete: this streams audio, not model tokens."""
    provider, connection, _ = _elevenlabs(_audio_message(b"\x01\x02", final=True))

    [c async for c in provider.stream("Hello there").chunks()]

    first = json.loads(connection.sent[0])
    assert first["text"] == "Hello there"
    assert first["xi_api_key"] == "test-key"
    assert json.loads(connection.sent[1]) == {"text": ""}


@pytest.mark.anyio
async def test_audio_chunks_are_decoded_in_order() -> None:
    provider, _, _ = _elevenlabs(
        _audio_message(b"\x01\x02"),
        _audio_message(b"\x03\x04"),
        _audio_message(b"\x05\x06", final=True),
    )

    chunks = [c async for c in provider.stream("Hello").chunks()]

    assert [chunk.audio for chunk in chunks] == [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
    assert chunks[-1].is_final


@pytest.mark.anyio
async def test_a_keepalive_without_audio_is_not_a_chunk() -> None:
    provider, _, _ = _elevenlabs(
        json.dumps({"audio": None}), _audio_message(b"\x01\x02", final=True)
    )

    assert len([c async for c in provider.stream("Hello").chunks()]) == 1


@pytest.mark.anyio
async def test_a_final_message_with_no_audio_still_ends_the_stream() -> None:
    provider, _, _ = _elevenlabs(
        _audio_message(b"\x01\x02"), json.dumps({"isFinal": True})
    )

    chunks = [c async for c in provider.stream("Hello").chunks()]

    assert len(chunks) == 2
    assert chunks[-1].is_final and chunks[-1].audio == b""


@pytest.mark.anyio
async def test_audio_that_is_not_base64_is_an_error() -> None:
    provider, _, _ = _elevenlabs(json.dumps({"audio": "not base64!!"}))

    with pytest.raises(VoiceError, match="base64"):
        [c async for c in provider.stream("Hello").chunks()]


@pytest.mark.anyio
async def test_an_odd_number_of_bytes_is_an_error() -> None:
    provider, _, _ = _elevenlabs(_audio_message(b"\x01\x02\x03"))

    with pytest.raises(VoiceError, match="odd number"):
        [c async for c in provider.stream("Hello").chunks()]


@pytest.mark.anyio
async def test_a_reported_error_is_raised_for_synthesis_too() -> None:
    provider, _, _ = _elevenlabs(json.dumps({"error": True, "message": "bad voice"}))

    with pytest.raises(VoiceError, match="bad voice"):
        [c async for c in provider.stream("Hello").chunks()]


@pytest.mark.anyio
async def test_a_disconnect_during_synthesis_is_unavailable() -> None:
    provider, _, _ = _elevenlabs(ConnectionResetError("gone"))

    with pytest.raises(VoiceUnavailable):
        [c async for c in provider.stream("Hello").chunks()]


def test_no_voice_is_refused_before_anything_opens() -> None:
    provider, _, _ = _elevenlabs(tts_voice="")

    with pytest.raises(VoiceError, match="No ElevenLabs voice"):
        provider.stream("Hello")


def test_no_synthesis_key_is_refused_before_anything_opens() -> None:
    provider, _, _ = _elevenlabs(elevenlabs_api_key="")

    with pytest.raises(VoiceUnavailable, match="No ElevenLabs API key"):
        provider.stream("Hello")


def test_a_rate_with_no_raw_output_is_refused_at_construction() -> None:
    with pytest.raises(VoiceError, match="no raw PCM output"):
        _elevenlabs(audio_sample_rate=44100)


@pytest.mark.anyio
async def test_closing_closes_the_synthesis_socket() -> None:
    provider, connection, _ = _elevenlabs(_audio_message(b"\x01\x02", final=True))
    stream = provider.stream("Hello")
    [c async for c in stream.chunks()]

    await stream.aclose()

    assert connection.closed
