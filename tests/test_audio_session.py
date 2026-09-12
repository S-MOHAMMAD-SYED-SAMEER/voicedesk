"""The adapter: audio in, dialogue in the middle, audio out.

What matters here is not that it works, but what it refuses to do when a piece
of it does not.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.audio import NOT_HEARD_REPLY, SPEECH_FAILURE_REPLY
from app.models import Appointment, Service, ToolCall, Turn
from app.providers.speech import read_wav
from app.providers.stt import SpeechUnavailable, Transcript
from app.providers.tts import VoiceUnavailable

from .conftest import FakeModel, FakeSTT, FakeTTS, say, spoke, use_tools

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


# --- the happy path -------------------------------------------------------


def test_the_transcript_reaches_the_dialogue_layer(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("Of course — what day suits you?"))
    session = voice(stt=FakeSTT("I'd like a haircut"), model=model)

    turn = session.speak(utterance)

    assert turn.transcript == "I'd like a haircut"
    caller_message = model.requests[0]["messages"][0]
    assert caller_message.content[0]["text"] == "I'd like a haircut"


def test_the_reply_reaches_the_synthesiser(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    tts = FakeTTS()
    session = voice(say("We're open until five."), tts=tts)

    turn = session.speak(utterance)

    assert tts.calls == ["We're open until five."]
    assert turn.reply == "We're open until five."


def test_playable_audio_comes_back(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    turn = voice(say("Certainly.")).speak(utterance)

    assert turn.speech is not None
    assert read_wav(turn.speech.audio).format.sample_rate == 16000
    assert turn.failed is False
    assert turn.failure is None


def test_the_audio_itself_is_handed_to_the_transcriber(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """Not a re-encoding of it, and not something the adapter made up."""
    stt = FakeSTT()
    voice(say("Certainly."), stt=stt).speak(utterance)

    assert stt.calls == [utterance]


def test_a_booking_made_by_voice_reaches_the_database(
    session: Session, voice, utterance, open_weekdays, haircut: Service
) -> None:
    audio_session = voice(
        use_tools(_check()), use_tools(_book()), say("You're booked in for ten.")
    )

    turn = audio_session.speak(utterance)

    appointment = session.execute(select(Appointment)).scalar_one()
    assert turn.dialogue is not None
    assert str(appointment.id) == turn.dialogue.booked_appointment_id


def test_confidence_is_carried_through(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    stt = FakeSTT(Transcript(text="A haircut please", confidence=0.62))

    turn = voice(say("Certainly."), stt=stt).speak(utterance)

    assert turn.confidence == 0.62


def test_low_confidence_changes_nothing_in_this_milestone(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """The "low twice in a row" rule needs endpoint detection. Not yet."""
    stt = FakeSTT(
        Transcript(text="mumble", confidence=0.05),
        Transcript(text="mumble again", confidence=0.05),
    )
    model = FakeModel(say("Sorry?"), say("Sorry?"))
    audio_session = voice(stt=stt, model=model)

    audio_session.speak(utterance)
    audio_session.speak(utterance)

    assert model.call_count == 2


def test_the_three_latencies_are_measured(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    turn = voice(say("Certainly.", latency_ms=30)).speak(utterance)

    assert turn.stt_latency_ms >= 0
    assert turn.tts_latency_ms >= 0
    assert turn.total_latency_ms >= 0
    assert turn.dialogue.llm_latency_ms == 30


def test_latency_is_reported_and_not_persisted(
    session: Session, voice, utterance, open_weekdays, haircut: Service
) -> None:
    """The turn columns stay null until the milestone that owns them."""
    voice(say("Certainly.")).speak(utterance)

    for turn in session.execute(select(Turn)).scalars().all():
        assert turn.audio_ms is None
        assert turn.stt_latency_ms is None
        assert turn.tts_latency_ms is None


# --- nothing heard ---------------------------------------------------------


def test_an_empty_transcript_never_reaches_the_model(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """Silence is not a question. Asking a model about it invites invention."""
    model = FakeModel(say("should never be used"))
    audio_session = voice(stt=FakeSTT(""), model=model)

    turn = audio_session.speak(utterance)

    assert model.call_count == 0
    assert turn.reply == NOT_HEARD_REPLY
    assert turn.failure == "empty"
    assert turn.failed is True


def test_a_whitespace_only_transcript_counts_as_nothing_heard(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("should never be used"))

    turn = voice(stt=FakeSTT("   \n "), model=model).speak(utterance)

    assert model.call_count == 0
    assert turn.failure == "empty"


def test_nothing_heard_still_gets_spoken_back(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    tts = FakeTTS()

    turn = voice(stt=FakeSTT(""), model=FakeModel(), tts=tts).speak(utterance)

    assert tts.calls == [NOT_HEARD_REPLY]
    assert turn.speech is not None


def test_nothing_heard_writes_no_transcript_rows(
    session: Session, voice, utterance, open_weekdays, haircut: Service
) -> None:
    """Nobody said anything, so there is nothing to record."""
    voice(stt=FakeSTT(""), model=FakeModel()).speak(utterance)

    assert session.execute(select(Turn)).first() is None


# --- the transcriber failed -------------------------------------------------


def test_a_transcriber_failure_never_reaches_the_model(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    model = FakeModel(say("should never be used"))
    stt = FakeSTT(raises=SpeechUnavailable("deepgram fell over"))

    turn = voice(stt=stt, model=model).speak(utterance)

    assert model.call_count == 0
    assert turn.failure == "stt"
    assert turn.failed is True


def test_a_transcriber_failure_fabricates_no_transcript(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """A broken provider must not look like the caller saying something."""
    stt = FakeSTT(raises=SpeechUnavailable("deepgram fell over"))

    turn = voice(model=FakeModel(), stt=stt).speak(utterance)

    assert turn.transcript == ""
    assert turn.reply == SPEECH_FAILURE_REPLY


def test_a_transcriber_failure_writes_no_transcript_rows(
    session: Session, voice, utterance, open_weekdays, haircut: Service
) -> None:
    stt = FakeSTT(raises=SpeechUnavailable("deepgram fell over"))

    voice(model=FakeModel(), stt=stt).speak(utterance)

    assert session.execute(select(Turn)).first() is None


def test_both_providers_down_is_reported_honestly(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """No audio, no transcript, and the failure named rather than hidden."""
    turn = voice(
        model=FakeModel(),
        stt=FakeSTT(raises=SpeechUnavailable("down")),
        tts=FakeTTS(raises=VoiceUnavailable("also down")),
    ).speak(utterance)

    assert turn.speech is None
    assert turn.failed is True
    assert turn.failure == "tts"
    assert turn.reply == SPEECH_FAILURE_REPLY


# --- the dialogue failed ----------------------------------------------------


def test_a_dialogue_failure_is_spoken_verbatim(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """Milestone 4 already said something safe. This layer does not improve on it."""
    from app.dialogue import MODEL_FAILURE_REPLY
    from app.providers.llm import ModelUnavailable

    tts = FakeTTS()
    audio_session = voice(raises=ModelUnavailable("the api fell over"), tts=tts)

    turn = audio_session.speak(utterance)

    assert turn.reply == MODEL_FAILURE_REPLY
    assert tts.calls == [MODEL_FAILURE_REPLY]
    assert turn.failure == "dialogue"
    assert turn.dialogue is not None and turn.dialogue.failed is True


def test_a_dialogue_failure_is_not_retried(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    from app.providers.llm import ModelUnavailable

    model = FakeModel(raises=ModelUnavailable("the api fell over"))

    voice(model=model).speak(utterance)

    assert model.call_count == 1


# --- the synthesiser failed -------------------------------------------------


def test_a_synthesiser_failure_keeps_the_text(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    tts = FakeTTS(raises=VoiceUnavailable("elevenlabs fell over"))

    turn = voice(say("We're open until five."), tts=tts).speak(utterance)

    assert turn.reply == "We're open until five."
    assert turn.speech is None
    assert turn.failure == "tts"
    assert turn.failed is True


def test_a_synthesiser_failure_never_re_runs_the_dialogue(
    session: Session, voice, utterance, open_weekdays, haircut: Service
) -> None:
    """Retrying for audio would risk booking the same caller twice."""
    model = FakeModel(
        use_tools(_check()), use_tools(_book()), say("You're booked in for ten.")
    )
    tts = FakeTTS(raises=VoiceUnavailable("elevenlabs fell over"))

    turn = voice(model=model, tts=tts).speak(utterance)

    assert len(session.execute(select(Appointment)).scalars().all()) == 1
    assert model.call_count == 3
    assert turn.dialogue is not None
    assert turn.dialogue.booked_appointment_id is not None


def test_a_synthesiser_failure_keeps_the_transcript(
    session: Session, voice, utterance, open_weekdays, haircut: Service
) -> None:
    """The dialogue happened. The recording of it is not conditional on audio."""
    tts = FakeTTS(raises=VoiceUnavailable("down"))

    voice(use_tools(_check()), say("Ten is free."), tts=tts).speak(utterance)

    assert len(session.execute(select(Turn)).scalars().all()) == 2
    assert session.execute(select(ToolCall)).first() is not None


# --- the greeting -----------------------------------------------------------


def test_the_greeting_is_synthesised_without_a_model_or_a_turn(
    session: Session, voice, open_weekdays, haircut: Service
) -> None:
    """Picking up the phone is not a conversational turn."""
    model = FakeModel()
    tts = FakeTTS()
    audio_session = voice(model=model, tts=tts)

    speech = audio_session.greeting("Thanks for calling. How can I help?")

    assert speech is not None
    assert tts.calls == ["Thanks for calling. How can I help?"]
    assert model.call_count == 0
    assert session.execute(select(Turn)).first() is None


def test_a_greeting_that_cannot_be_spoken_is_not_fatal(
    voice, open_weekdays, haircut: Service
) -> None:
    audio_session = voice(model=FakeModel(), tts=FakeTTS(raises=VoiceUnavailable("x")))

    assert audio_session.greeting("Thanks for calling.") is None


# --- what a turn consumed (milestone 8) -----------------------------------


def test_a_turn_reports_the_audio_the_recogniser_was_given(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    session = voice(say("Of course."), stt=FakeSTT(spoke(audio_ms=1500)))

    turn = session.speak(utterance)

    assert turn.stt_audio_ms == 1500
    assert turn.stt_provider == "offline"


def test_a_turn_reports_the_characters_the_synthesiser_was_given(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    session = voice(say("Of course."))

    turn = session.speak(utterance)

    assert turn.tts_characters == len("Of course.")
    assert turn.tts_provider == "fake"


def test_a_recogniser_that_reported_no_duration_leaves_it_null(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """Null is "we were not told", and must not become a zero."""
    session = voice(say("Of course."), stt=FakeSTT(spoke(audio_ms=None)))

    turn = session.speak(utterance)

    assert turn.stt_audio_ms is None


def test_a_turn_that_heard_nothing_still_reports_what_listening_took(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    session = voice(say("Unused."), stt=FakeSTT(spoke("   ", audio_ms=900)))

    turn = session.speak(utterance)

    assert turn.reply == NOT_HEARD_REPLY
    assert turn.stt_audio_ms == 900


def test_a_broken_recogniser_reports_no_usage_at_all(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    """Nothing was measured, so nothing is claimed."""
    session = voice(say("Unused."), stt=FakeSTT(raises=SpeechUnavailable("down")))

    turn = session.speak(utterance)

    assert turn.failure == "stt"
    assert turn.stt_audio_ms is None
    assert turn.stt_provider == ""


def test_a_broken_synthesiser_reports_no_characters(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    session = voice(say("Of course."), tts=FakeTTS(raises=VoiceUnavailable("down")))

    turn = session.speak(utterance)

    assert turn.failure == "tts"
    assert turn.tts_characters is None
    assert turn.tts_provider == ""


def test_the_audio_layer_puts_no_price_on_any_of_it(
    voice, utterance, open_weekdays, haircut: Service
) -> None:
    session = voice(say("Of course."))

    turn = session.speak(utterance)

    assert not [field for field in vars(turn) if "cost" in field or "usd" in field]
