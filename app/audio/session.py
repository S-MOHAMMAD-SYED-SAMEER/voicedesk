"""The audio adapter around the dialogue layer.

    audio → speech-to-text → Conversation.send → text-to-speech → audio

That is the whole job. `VoiceSession` holds no conversation state, runs no
tools, knows nothing about the calendar and never speaks to a model. The
specification is blunt about why: "Audio is an adapter, not the product." If
anything in this module started deciding what to say, there would be two
dialogue layers and they would eventually disagree.

Milestone 5 is utterance-based. One complete recording arrives, one complete
reply goes back. There is no voice-activity detection, no partial transcript
and no barge-in, and no claim is made about the specification's 1.2-second
target — press-to-talk does not meet it and is not measured against it.

Three failures matter here, and each has one correct answer:

* **Nothing was heard.** The model is not called at all. Asking a language
  model what to say about silence costs money and invites invention.
* **The transcriber failed.** The model is not called either, and no
  transcript is fabricated. Silence and a broken provider must never look the
  same.
* **The synthesiser failed.** The dialogue already happened and is already
  persisted — including any booking it made. It is not run again. The reply
  comes back as text with no audio, because retrying would risk booking the
  same caller twice.
"""

import time
from dataclasses import dataclass

from app.audio.errors import AudioError
from app.audio.format import validate_utterance
from app.config import Settings, get_settings
from app.dialogue import Conversation, DialogueResult
from app.providers.speech import Audio
from app.providers.stt import SpeechError, SpeechToText, Transcript
from app.providers.tts import Speech, TextToSpeech, VoiceError

# Fixed lines, spoken when there is nothing a model should be asked about.
NOT_HEARD_REPLY = "Sorry, I didn't catch that — could you say that again?"
SPEECH_FAILURE_REPLY = (
    "I'm sorry, I'm having trouble hearing you. Let me pass you to a colleague."
)


@dataclass(frozen=True)
class VoiceTurn:
    """One press-to-talk exchange: what was heard, said, and how long it took.

    Nothing here is written to a database by this layer. The latencies and the
    usage are reported outwards and a transport writes them, which is what
    keeps the audio layer free of SQL: it measures, and something above it
    decides what to keep.

    The greeting is deliberately not accounted for. It is synthesised before
    anybody has said anything, so there is no turn row to attach it to, and
    inventing one to hold an accounting figure would put a sentence in the
    transcript that nobody said. It is therefore a small, known under-count.
    """

    transcript: str
    reply: str
    speech: Speech | None
    dialogue: DialogueResult | None
    stt_latency_ms: int = 0
    tts_latency_ms: int = 0
    total_latency_ms: int = 0
    confidence: float | None = None
    failed: bool = False
    # "audio" | "stt" | "empty" | "dialogue" | "tts" | None
    failure: str | None = None
    # What the two speech providers were given, carried for whoever accounts
    # for it. Plain numbers and plain names: this layer measures, and has no
    # idea what any of it costs. Null where a provider reported nothing —
    # never zero, which would claim it was handed nothing at all.
    stt_provider: str = ""
    stt_audio_ms: int | None = None
    tts_provider: str = ""
    tts_characters: int | None = None


class VoiceSession:
    """Wraps one `Conversation` in a microphone and a speaker."""

    def __init__(
        self,
        conversation: Conversation,
        stt: SpeechToText,
        tts: TextToSpeech,
        settings: Settings | None = None,
    ) -> None:
        self._conversation = conversation
        self._stt = stt
        self._tts = tts
        self._settings = settings or get_settings()

    def greeting(self, text: str) -> Speech | None:
        """Say the opening line.

        A greeting is not a turn: nobody said anything to prompt it, no model
        was asked and no tool ran, so it writes no transcript rows. It is the
        receptionist picking up the phone.
        """
        try:
            return self._tts.synthesize(text)
        except VoiceError:
            return None

    def speak(self, audio: Audio) -> VoiceTurn:
        """One caller utterance, all the way to audio coming back."""
        started = time.perf_counter()

        try:
            transcript = self._stt.transcribe(audio)
        except SpeechError:
            return self._say(
                SPEECH_FAILURE_REPLY,
                started=started,
                failure="stt",
                transcript="",
            )
        stt_latency_ms = int((time.perf_counter() - started) * 1000)

        if not transcript.text.strip():
            # Deliberately before the model: silence is not a question.
            # Nothing was said, but the recogniser still listened to it and
            # will still be billed for it.
            return self._say(
                NOT_HEARD_REPLY,
                started=started,
                failure="empty",
                transcript="",
                stt_latency_ms=stt_latency_ms,
                confidence=transcript.confidence,
                heard=transcript,
            )

        result = self._conversation.send(transcript.text)

        return self._say(
            result.text,
            started=started,
            # The dialogue layer has already said something safe; this layer
            # does not add a second apology or a different outcome.
            failure="dialogue" if result.failed else None,
            transcript=transcript.text,
            stt_latency_ms=stt_latency_ms,
            confidence=transcript.confidence,
            dialogue=result,
            heard=transcript,
        )

    # --- internals ---------------------------------------------------------

    def _say(
        self,
        reply: str,
        *,
        started: float,
        failure: str | None,
        transcript: str,
        stt_latency_ms: int = 0,
        confidence: float | None = None,
        dialogue: DialogueResult | None = None,
        heard: Transcript | None = None,
    ) -> VoiceTurn:
        """Synthesise the reply, and report honestly if that is all that worked."""
        speech: Speech | None = None
        tts_started = time.perf_counter()
        try:
            speech = self._tts.synthesize(reply)
        except VoiceError:
            # Whatever happened before this stands. In particular a booking
            # that went through stays gone through: the dialogue is never
            # re-run to get a second try at the audio.
            failure = "tts"
        tts_latency_ms = int((time.perf_counter() - tts_started) * 1000)

        return VoiceTurn(
            transcript=transcript,
            reply=reply,
            speech=speech,
            dialogue=dialogue,
            stt_latency_ms=stt_latency_ms,
            tts_latency_ms=tts_latency_ms,
            total_latency_ms=int((time.perf_counter() - started) * 1000),
            confidence=confidence,
            failed=failure is not None,
            failure=failure,
            stt_provider=heard.provider_name if heard is not None else "",
            stt_audio_ms=heard.audio_ms if heard is not None else None,
            tts_provider=speech.provider_name if speech is not None else "",
            # One number for one request, straight from the provider. The
            # streaming path has to be more careful; see `app/realtime`.
            tts_characters=speech.characters if speech is not None else None,
        )


def utterance_from_bytes(data: bytes, settings: Settings | None = None) -> Audio:
    """The edge: raw frame bytes, checked against the milestone-5 contract."""
    resolved = settings or get_settings()
    return validate_utterance(
        data,
        sample_rate=resolved.audio_sample_rate,
        max_bytes=resolved.max_utterance_bytes,
    )


__all__ = [
    "NOT_HEARD_REPLY",
    "SPEECH_FAILURE_REPLY",
    "AudioError",
    "VoiceSession",
    "VoiceTurn",
    "utterance_from_bytes",
]
