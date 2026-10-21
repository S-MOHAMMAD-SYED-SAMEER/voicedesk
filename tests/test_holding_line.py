"""The holding line: said once if the model is slow, never cut off, never part of the conversation."""


import anyio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.audio.holding import with_holding_line
from app.config import get_settings
from app.models import Turn
from app.providers.offline_streaming import OfflineStreamingTextToSpeech

from .conftest import FakeModel, FakeSTT, FakeTTS, SlowModel, say, wav_bytes

# --- the helper --------------------------------------------------------------


@pytest.mark.anyio
async def test_fast_work_never_says_the_line() -> None:
    said: list[str] = []

    async def work() -> str:
        return "done"

    async def hold() -> None:
        said.append("hold")

    assert await with_holding_line(work, hold, after_seconds=0.2) == "done"
    assert said == []


@pytest.mark.anyio
async def test_slow_work_says_the_line_exactly_once() -> None:
    said: list[str] = []

    async def work() -> str:
        await anyio.sleep(0.35)
        return "done"

    async def hold() -> None:
        said.append("hold")

    assert await with_holding_line(work, hold, after_seconds=0.05) == "done"
    assert said == ["hold"]


@pytest.mark.anyio
async def test_a_line_that_has_started_is_allowed_to_finish() -> None:
    events: list[str] = []

    async def work() -> str:
        await anyio.sleep(0.12)
        events.append("work done")
        return "done"

    async def hold() -> None:
        events.append("hold start")
        await anyio.sleep(0.3)  # longer than the rest of the work
        events.append("hold end")

    await with_holding_line(work, hold, after_seconds=0.05)

    assert events == ["hold start", "work done", "hold end"]


@pytest.mark.anyio
async def test_the_works_own_exception_comes_out_unwrapped() -> None:
    async def work() -> str:
        raise ValueError("boom")

    async def hold() -> None:
        return None

    with pytest.raises(ValueError, match="boom"):
        await with_holding_line(work, hold, after_seconds=0.05)


@pytest.mark.anyio
async def test_a_holding_line_that_fails_does_not_fail_the_turn() -> None:
    async def work() -> str:
        await anyio.sleep(0.2)
        return "done"

    async def hold() -> None:
        raise RuntimeError("speech provider down")

    assert await with_holding_line(work, hold, after_seconds=0.05) == "done"


# --- the realtime session ----------------------------------------------------


def _after(settings, seconds: float):
    return settings.model_copy(update={"holding_line_after_seconds": seconds})


async def _say_something(session) -> None:
    from .conftest import pcm_silence, pcm_tone

    for _ in range(10):
        await session.feed(pcm_tone(9000))
    for _ in range(40):
        await session.feed(pcm_silence())


@pytest.mark.anyio
async def test_realtime_says_the_line_first_when_the_model_is_slow(
    realtime, realtime_settings, open_weekdays, haircut
) -> None:
    tts = OfflineStreamingTextToSpeech(sample_rate=8000)
    session, sink = realtime(
        model=SlowModel(say("We close at five."), seconds=0.5),
        tts=tts,
        settings=_after(realtime_settings, 0.1),
    )

    await _say_something(session)

    assert tts.spoken == ["One moment, let me check.", "We close at five."]
    assert sink.chunks


@pytest.mark.anyio
async def test_realtime_says_nothing_extra_when_the_model_is_quick(
    realtime, realtime_settings, open_weekdays, haircut
) -> None:
    tts = OfflineStreamingTextToSpeech(sample_rate=8000)
    session, _ = realtime(
        say("We close at five."), tts=tts, settings=_after(realtime_settings, 0.5)
    )

    await _say_something(session)

    assert tts.spoken == ["We close at five."]


@pytest.mark.anyio
async def test_the_holding_line_is_not_written_to_the_transcript(
    db_session: Session, realtime, realtime_settings, open_weekdays, haircut
) -> None:
    session, _ = realtime(
        model=SlowModel(say("We close at five."), seconds=0.4),
        settings=_after(realtime_settings, 0.1),
    )

    await _say_something(session)

    texts = [row.text for row in db_session.execute(select(Turn)).scalars()]
    assert not any("One moment" in text for text in texts)
    assert "We close at five." in texts


@pytest.mark.anyio
async def test_the_holding_line_is_configurable(
    realtime, realtime_settings, open_weekdays, haircut
) -> None:
    tts = OfflineStreamingTextToSpeech(sample_rate=8000)
    settings = _after(realtime_settings, 0.1).model_copy(
        update={"holding_line": "Bear with me a second."}
    )
    session, _ = realtime(
        model=SlowModel(say("Done."), seconds=0.4), tts=tts, settings=settings
    )

    await _say_something(session)

    assert tts.spoken[0] == "Bear with me a second."


# --- the push-to-talk harness -------------------------------------------------


@pytest.fixture
def slow_harness(migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch):
    from app.main import create_app

    def build(model, tts=None):
        monkeypatch.setenv("VOICEDESK_HOLDING_LINE_AFTER_SECONDS", "0.1")
        get_settings.cache_clear()
        monkeypatch.setattr("app.api.harness.build_model", lambda settings=None: model)
        monkeypatch.setattr("app.api.harness.build_stt", lambda settings=None: FakeSTT("When do you close?"))
        speech = tts or FakeTTS()
        monkeypatch.setattr("app.api.harness.build_tts", lambda settings=None: speech)
        return TestClient(create_app()), speech

    return build


def test_the_harness_plays_the_line_before_the_reply_when_the_model_is_slow(
    slow_harness, open_weekdays, haircut
) -> None:
    client, tts = slow_harness(SlowModel(say("We close at five."), seconds=0.5))

    with client.websocket_connect("/ws/harness") as socket:
        socket.receive_json()
        socket.receive_bytes()
        socket.send_bytes(wav_bytes())
        first = socket.receive_json()
        holding_audio = socket.receive_bytes()
        turn = socket.receive_json()
        reply_audio = socket.receive_bytes()

    assert first == {"type": "holding", "text": "One moment, let me check."}
    assert holding_audio and reply_audio
    assert turn["type"] == "turn" and turn["reply"] == "We close at five."
    assert tts.calls[1:] == ["One moment, let me check.", "We close at five."]


def test_the_harness_says_nothing_extra_when_the_model_is_quick(
    slow_harness, open_weekdays, haircut, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, tts = slow_harness(FakeModel(say("We close at five.")))
    monkeypatch.setenv("VOICEDESK_HOLDING_LINE_AFTER_SECONDS", "5")
    get_settings.cache_clear()

    with client.websocket_connect("/ws/harness") as socket:
        socket.receive_json()
        socket.receive_bytes()
        socket.send_bytes(wav_bytes())
        first = socket.receive_json()
        socket.receive_bytes()

    assert first["type"] == "turn"
    assert tts.calls[1:] == ["We close at five."]
