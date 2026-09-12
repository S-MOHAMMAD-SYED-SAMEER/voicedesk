"""Interrupting the receptionist, and what must not survive it.

Cancellation here is partial by nature: a thread running a model request and
the tool calls it made cannot be killed, only abandoned. So the guarantee is
not "the work stopped" — it is "nothing stale is ever heard, and nothing is
ever done twice".
"""

import anyio
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Appointment, Service, Turn
from app.providers.offline_streaming import (
    OfflineStreamingSpeechToText,
    OfflineStreamingTextToSpeech,
)
from app.providers.speech import PCM_S16LE, AudioFormat
from app.providers.streaming_tts import SpeechChunk
from app.realtime.generation import Generations

from .conftest import FakeModel, RecordingSink, pcm_silence, pcm_tone, say, use_tools

LOUD = pcm_tone(9000)
QUIET = pcm_silence()
MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"
FORMAT = AudioFormat(PCM_S16LE, 8000, 1, "raw")


def _book():
    return (
        "book_appointment",
        {
            "service_name": "Haircut",
            "starts_at": TEN,
            "customer_name": "Ada Lovelace",
            "phone": "+447700900123",
        },
    )


class SlowVoiceStream:
    """Synthesis that yields a chunk at a time, waiting to be let go.

    Lets a test stand exactly in the middle of a reply and interrupt it.
    """

    def __init__(self, text: str, audio_format: AudioFormat, gate: anyio.Event) -> None:
        self._text = text
        self._format = audio_format
        self._gate = gate
        self.closed = False
        self.produced = 0

    async def chunks(self):
        for index in range(5):
            if self.closed:
                return
            self.produced += 1
            yield SpeechChunk(
                audio=b"\x01\x02" * 80,
                format=self._format,
                is_final=index == 4,
            )
            if index == 0:
                # Hold here until the test says otherwise.
                await self._gate.wait()

    async def aclose(self) -> None:
        self.closed = True
        self._gate.set()


class SlowTTS:
    """Opens `SlowVoiceStream`s, one per reply."""

    def __init__(self, sample_rate: int = 8000) -> None:
        self._format = AudioFormat(PCM_S16LE, sample_rate, 1, "raw")
        self.gate = anyio.Event()
        self.streams: list[SlowVoiceStream] = []

    @property
    def format(self) -> AudioFormat:
        return self._format

    def stream(self, text: str, voice: str | None = None) -> SlowVoiceStream:
        opened = SlowVoiceStream(text, self._format, self.gate)
        self.streams.append(opened)
        return opened


async def _say_something(session, speech: int = 10, silence: int = 40) -> None:
    for _ in range(speech):
        await session.feed(LOUD)
    for _ in range(silence):
        await session.feed(QUIET)


# --- the generation counter -----------------------------------------------


def test_a_generation_is_current_until_the_next_one() -> None:
    generations = Generations()
    first = generations.token(generations.next())

    assert first.current
    generations.next()
    assert first.stale


def test_generations_only_go_forward() -> None:
    generations = Generations()

    assert [generations.next() for _ in range(3)] == [1, 2, 3]


# --- interrupting ---------------------------------------------------------


@pytest.mark.anyio
async def test_speech_during_a_reply_interrupts_it(
    realtime, open_weekdays, haircut: Service
) -> None:
    tts = SlowTTS()
    session, sink = realtime(say("A long answer indeed."), tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        # The reply is mid-flight, held at its first chunk.
        await anyio.sleep(0.05)
        assert session.playing

        for _ in range(10):
            await session.feed(LOUD)

        assert sink.clears == 1
        assert tts.streams[0].closed
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_interrupting_bumps_the_generation(
    realtime, open_weekdays, haircut: Service
) -> None:
    tts = SlowTTS()
    session, _ = realtime(say("A long answer indeed."), tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        before = session.generation

        for _ in range(10):
            await session.feed(LOUD)

        assert session.generation > before
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_the_interrupted_turn_is_marked_as_such(
    realtime, open_weekdays, haircut: Service
) -> None:
    tts = SlowTTS()
    session, _ = realtime(say("A long answer indeed."), tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        for _ in range(10):
            await session.feed(LOUD)

        assert session.turns[0].interrupted
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_queued_audio_is_discarded_not_merely_stopped(
    realtime, open_weekdays, haircut: Service
) -> None:
    """A caller who interrupts must not hear the rest of the sentence."""
    tts = SlowTTS()
    session, sink = realtime(say("A long answer indeed."), tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        assert sink.chunks

        for _ in range(10):
            await session.feed(LOUD)

        assert sink.chunks == []
        assert "clear" in sink.order
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_no_further_audio_is_sent_after_the_interruption(
    realtime, open_weekdays, haircut: Service
) -> None:
    """The check is on every chunk, because a provider has some in flight."""
    tts = SlowTTS()
    session, sink = realtime(say("A long answer indeed."), tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        for _ in range(10):
            await session.feed(LOUD)
        after_clear = len(sink.order)

        # Let the held synthesis run on, exactly as a real one would.
        tts.gate.set()
        await anyio.sleep(0.05)

        assert sink.order[after_clear:].count("send") == 0
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_a_stale_chunk_is_dropped_rather_than_queued(
    realtime, open_weekdays, haircut: Service
) -> None:
    from app.realtime.session import TurnTiming

    session, sink = realtime(say("Hello."))
    generation = session._generations.token(session._generations.next())
    session._generations.next()  # something else happened since

    await session._speak("stale words", generation, TurnTiming())

    assert sink.chunks == []


@pytest.mark.anyio
async def test_barge_in_can_be_switched_off(
    realtime, realtime_settings, open_weekdays, haircut: Service
) -> None:
    tts = SlowTTS()
    session, sink = realtime(
        say("A long answer indeed."),
        tts=tts,
        settings=realtime_settings.model_copy(update={"barge_in_enabled": False}),
    )

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        for _ in range(10):
            await session.feed(LOUD)

        assert sink.clears == 0
        assert not tts.streams[0].closed
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_speech_when_nothing_is_playing_is_not_an_interruption(
    realtime, open_weekdays, haircut: Service
) -> None:
    session, sink = realtime(say("One."), say("Two."))

    await _say_something(session)
    await _say_something(session)

    assert sink.clears == 0
    assert len(session.turns) == 2


# --- what survives an interruption ----------------------------------------


@pytest.mark.anyio
async def test_a_booking_that_already_happened_is_not_undone(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    """Deliberate, and documented.

    A thread cannot be killed, so a tool that already committed stays
    committed. Undoing a real booking because the caller started talking
    would be worse than letting it stand; what is discarded is the reply.
    """
    from app.dialogue import Conversation
    from app.realtime import RealtimeSession

    model = FakeModel(
        use_tools(("check_availability", {"service_name": "Haircut", "day": MONDAY})),
        use_tools(_book()),
        say("You're booked in for ten."),
    )
    tts = SlowTTS()
    session, sink = realtime(model=model, tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        for _ in range(10):
            await session.feed(LOUD)
        tts.gate.set()
        await anyio.sleep(0.05)
        group.cancel_scope.cancel()

    assert len(db_session.execute(select(Appointment)).scalars().all()) == 1
    assert sink.chunks == []


@pytest.mark.anyio
async def test_the_dialogue_is_never_run_twice(
    realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("A long answer indeed."), say("A second answer."))
    tts = SlowTTS()
    session, _ = realtime(model=model, tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        for _ in range(10):
            await session.feed(LOUD)
        tts.gate.set()
        await anyio.sleep(0.05)

        # One model call for the interrupted turn; the interruption itself
        # has not finished being spoken, so it has not asked yet.
        assert model.call_count == 1
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_the_transcript_of_an_interrupted_turn_still_stands(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    """It is a record of what happened, not of what was heard."""
    tts = SlowTTS()
    session, _ = realtime(say("A long answer indeed."), tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        for _ in range(10):
            await session.feed(LOUD)
        group.cancel_scope.cancel()

    assert len(db_session.execute(select(Turn)).scalars().all()) == 2


@pytest.mark.anyio
async def test_a_new_turn_follows_the_interruption(
    realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("A long answer indeed."), say("Of course."))
    tts = SlowTTS()
    session, _ = realtime(model=model, tts=tts)

    async with anyio.create_task_group() as group:
        session.attach(group)
        await _say_something(session)
        await anyio.sleep(0.05)
        await _say_something(session)  # interrupt, then finish speaking
        await anyio.sleep(0.05)

        assert len(session.turns) == 2
        group.cancel_scope.cancel()


@pytest.mark.anyio
async def test_a_final_transcript_from_a_stale_turn_is_discarded(
    realtime, open_weekdays, haircut: Service
) -> None:
    """Recognition can finish after the caller has already moved on."""
    model = FakeModel(say("should never be used"))
    session, _ = realtime(model=model, stt=OfflineStreamingSpeechToText())
    generation = session._generations.token(session._generations.next())
    session._generations.next()

    from app.realtime.session import RealtimeTurn

    turn = RealtimeTurn(generation=generation.number)
    await session._transcribe_and_answer(b"\x00\x00" * 800, generation, turn)

    assert model.call_count == 0
