"""The realtime session: audio in, the same dialogue, audio out."""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Appointment, Service, Turn
from app.providers.offline_streaming import (
    OfflineStreamingSpeechToText,
    OfflineStreamingTextToSpeech,
)
from app.providers.streaming_stt import SpeechUnavailable
from app.providers.streaming_tts import VoiceUnavailable
from app.realtime import NOT_HEARD_REPLY, SPEECH_FAILURE_REPLY

from .conftest import FakeModel, pcm_silence, pcm_tone, say, use_tools

LOUD = pcm_tone(9000)
QUIET = pcm_silence()
MONDAY = "2026-03-02"
TEN = "2026-03-02T10:00:00+00:00"


def _check():
    return ("check_availability", {"service_name": "Haircut", "day": MONDAY})


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


async def _say_something(session, speech: int = 10, silence: int = 40) -> None:
    """Speak, then stop, so the endpointer closes a turn."""
    for _ in range(speech):
        await session.feed(LOUD)
    for _ in range(silence):
        await session.feed(QUIET)


# --- the happy path -------------------------------------------------------


@pytest.mark.anyio
async def test_a_final_transcript_reaches_the_dialogue(
    realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("Of course — what day suits you?"))
    session, _ = realtime(model=model)

    await _say_something(session)

    assert model.call_count == 1
    spoken = model.requests[0]["messages"][0].content[0]["text"]
    assert spoken == "I'd like to book an appointment"


@pytest.mark.anyio
async def test_the_reply_is_synthesised_and_sent(
    realtime, open_weekdays, haircut: Service
) -> None:
    tts = OfflineStreamingTextToSpeech(sample_rate=8000)
    session, sink = realtime(say("We're open until five."), tts=tts)

    await _say_something(session)

    assert tts.spoken == ["We're open until five."]
    assert sink.chunks
    assert sink.chunks[-1].is_final
    assert sink.marks


@pytest.mark.anyio
async def test_the_transcript_is_recorded(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    session, _ = realtime(say("We're open until five."))

    await _say_something(session)

    turns = db_session.execute(select(Turn).order_by(Turn.created_at)).scalars().all()
    assert [turn.text for turn in turns] == [
        "I'd like to book an appointment",
        "We're open until five.",
    ]


@pytest.mark.anyio
async def test_a_booking_made_in_realtime_reaches_the_database(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    session, _ = realtime(
        use_tools(_check()), use_tools(_book()), say("You're booked in for ten.")
    )

    await _say_something(session)

    appointment = db_session.execute(select(Appointment)).scalar_one()
    assert appointment.customer_name == "Ada Lovelace"


@pytest.mark.anyio
async def test_partials_are_reported_and_nothing_more(
    realtime, open_weekdays, haircut: Service
) -> None:
    seen: list[str] = []

    async def remember(partial) -> None:
        seen.append(partial.text)

    session, _ = realtime(say("Certainly."), on_partial=remember)

    await _say_something(session)

    assert seen == ["I'd like", "I'd like to book"]


@pytest.mark.anyio
async def test_a_partial_never_reaches_the_dialogue(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    """A tool called from a guess is a booking made from a guess.

    The model is asked exactly once, with the final text — never with either
    of the two partials that preceded it.
    """
    model = FakeModel(say("Certainly."))
    session, _ = realtime(model=model)

    await _say_something(session)

    assert model.call_count == 1
    asked = [
        message.content[0]["text"]
        for message in model.requests[0]["messages"]
        if message.role == "user"
    ]
    assert asked == ["I'd like to book an appointment"]
    assert "I'd like to book" not in asked


@pytest.mark.anyio
async def test_several_turns_share_one_conversation(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("Of course."), say("Nine or ten."), say("Booked."))
    session, _ = realtime(model=model)

    for _ in range(3):
        await _say_something(session)

    assert model.call_count == 3
    assert len(db_session.execute(select(Turn)).scalars().all()) == 6
    assert len(model.requests[-1]["messages"]) == 5


@pytest.mark.anyio
async def test_each_turn_takes_the_next_generation(
    realtime, open_weekdays, haircut: Service
) -> None:
    session, _ = realtime(say("One."), say("Two."))

    await _say_something(session)
    await _say_something(session)

    assert [turn.generation for turn in session.turns] == [1, 2]


# --- the greeting ---------------------------------------------------------


@pytest.mark.anyio
async def test_the_greeting_is_spoken_and_is_not_a_turn(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel()
    tts = OfflineStreamingTextToSpeech(sample_rate=8000)
    session, sink = realtime(model=model, tts=tts)

    await session.greeting("Thanks for calling.")

    assert tts.spoken == ["Thanks for calling."]
    assert sink.chunks
    assert model.call_count == 0
    assert db_session.execute(select(Turn)).first() is None
    assert session.turns == []


# --- nothing heard --------------------------------------------------------


@pytest.mark.anyio
async def test_silence_never_reaches_the_model(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    """No utterance, no recognition, no model call, no row."""
    model = FakeModel(say("should never be used"))
    session, _ = realtime(model=model)

    for _ in range(200):
        await session.feed(QUIET)

    assert model.call_count == 0
    assert session.turns == []
    assert db_session.execute(select(Turn)).first() is None


@pytest.mark.anyio
async def test_an_empty_transcript_never_reaches_the_model(
    realtime, open_weekdays, haircut: Service
) -> None:
    """Recognition heard nothing. That is not a question for a model."""
    model = FakeModel(say("should never be used"))
    session, _ = realtime(
        model=model, stt=OfflineStreamingSpeechToText(partials=(), final="   ")
    )

    await _say_something(session)

    assert model.call_count == 0
    assert session.turns[-1].failure == "empty"
    assert session.turns[-1].reply == NOT_HEARD_REPLY


# --- the transcriber failed ------------------------------------------------


@pytest.mark.anyio
async def test_a_transcriber_failure_never_reaches_the_model(
    realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("should never be used"))
    session, _ = realtime(
        model=model,
        stt=OfflineStreamingSpeechToText(raises=SpeechUnavailable("gone")),
    )

    await _say_something(session)

    assert model.call_count == 0
    assert session.turns[-1].failure == "stt"
    assert session.turns[-1].transcript == ""


@pytest.mark.anyio
async def test_a_transcriber_failure_still_says_something_safe(
    realtime, open_weekdays, haircut: Service
) -> None:
    tts = OfflineStreamingTextToSpeech(sample_rate=8000)
    session, sink = realtime(
        model=FakeModel(),
        stt=OfflineStreamingSpeechToText(raises=SpeechUnavailable("gone")),
        tts=tts,
    )

    await _say_something(session)

    assert tts.spoken == [SPEECH_FAILURE_REPLY]
    assert sink.chunks


# --- the dialogue failed ---------------------------------------------------


@pytest.mark.anyio
async def test_a_dialogue_failure_is_spoken_verbatim(
    realtime, open_weekdays, haircut: Service
) -> None:
    """Milestone 4 already said something safe; this layer does not improve it."""
    from app.dialogue import MODEL_FAILURE_REPLY
    from app.providers.llm import ModelUnavailable

    tts = OfflineStreamingTextToSpeech(sample_rate=8000)
    session, _ = realtime(raises=ModelUnavailable("the api fell over"), tts=tts)

    await _say_something(session)

    assert tts.spoken == [MODEL_FAILURE_REPLY]
    assert session.turns[-1].failure == "dialogue"


@pytest.mark.anyio
async def test_a_dialogue_failure_is_not_retried(
    realtime, open_weekdays, haircut: Service
) -> None:
    from app.providers.llm import ModelUnavailable

    model = FakeModel(raises=ModelUnavailable("the api fell over"))
    session, _ = realtime(model=model)

    await _say_something(session)

    assert model.call_count == 1


# --- the synthesiser failed ------------------------------------------------


@pytest.mark.anyio
async def test_a_synthesiser_failure_never_re_runs_the_dialogue(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    """Retrying for audio would risk booking the same caller twice."""
    model = FakeModel(use_tools(_check()), use_tools(_book()), say("Booked."))
    session, _ = realtime(
        model=model,
        tts=OfflineStreamingTextToSpeech(raises=VoiceUnavailable("down")),
    )

    await _say_something(session)

    assert len(db_session.execute(select(Appointment)).scalars().all()) == 1
    assert model.call_count == 3


@pytest.mark.anyio
async def test_a_synthesiser_failure_keeps_the_transcript(
    db_session: Session, realtime, open_weekdays, haircut: Service
) -> None:
    session, sink = realtime(
        say("We're open until five."),
        tts=OfflineStreamingTextToSpeech(raises=VoiceUnavailable("down")),
    )

    await _say_something(session)

    assert len(db_session.execute(select(Turn)).scalars().all()) == 2
    assert sink.chunks == []
    assert session.turns[-1].reply == "We're open until five."


# --- ending mid-sentence ---------------------------------------------------


@pytest.mark.anyio
async def test_finishing_answers_a_half_finished_sentence(
    realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("Of course."))
    session, _ = realtime(model=model)
    for _ in range(10):
        await session.feed(LOUD)

    await session.finish()

    assert model.call_count == 1


@pytest.mark.anyio
async def test_finishing_with_nothing_said_does_nothing(
    realtime, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("should never be used"))
    session, _ = realtime(model=model)

    await session.finish()

    assert model.call_count == 0
