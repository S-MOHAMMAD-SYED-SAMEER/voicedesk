import os
from collections.abc import Iterator

import pytest
import sqlalchemy
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import reset_engine
from app.runtime import reset_admission
from app.main import create_app

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tests never touch the development database. Override with
# VOICEDESK_TEST_DATABASE_URL to point at a different server.
TEST_DATABASE_URL = os.environ.get(
    "VOICEDESK_TEST_DATABASE_URL",
    "postgresql+psycopg://voicedesk:voicedesk@localhost:5432/voicedesk_test",
)


@pytest.fixture
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give a test a clean settings/engine cache and restore it afterwards."""
    monkeypatch.setenv("VOICEDESK_ENVIRONMENT", "test")
    get_settings.cache_clear()
    reset_engine()
    reset_admission()
    try:
        yield
    finally:
        get_settings.cache_clear()
        reset_engine()
        reset_admission()


@pytest.fixture
def client(settings_env: None) -> Iterator[TestClient]:
    yield TestClient(create_app())


@pytest.fixture(scope="session")
def database_url() -> str:
    """The test database URL, skipping the test if no server is reachable."""
    engine = create_engine(TEST_DATABASE_URL)
    try:
        with engine.connect():
            pass
    except sqlalchemy.exc.OperationalError as exc:
        pytest.skip(f"no PostgreSQL at {TEST_DATABASE_URL}: {exc}")
    finally:
        engine.dispose()
    return TEST_DATABASE_URL


@pytest.fixture
def alembic_config(database_url: str) -> AlembicConfig:
    config = AlembicConfig(os.path.join(PROJECT_ROOT, "alembic.ini"))
    config.set_main_option("script_location", os.path.join(PROJECT_ROOT, "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


@pytest.fixture
def migrated_engine(
    database_url: str,
    alembic_config: AlembicConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Engine]:
    """A database migrated to head, torn back down to empty afterwards.

    Running the real migration rather than `Base.metadata.create_all` is the
    point: the exclusion constraint that prevents double booking exists only
    in the migration, so a schema built any other way would not have it.
    """
    monkeypatch.setenv("VOICEDESK_DATABASE_URL", database_url)
    get_settings.cache_clear()
    reset_engine()

    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")

    engine = create_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()
        command.downgrade(alembic_config, "base")
        get_settings.cache_clear()
        reset_engine()


@pytest.fixture
def session(migrated_engine: Engine) -> Iterator[Session]:
    with Session(migrated_engine) as db_session:
        yield db_session


# --- calendar fixtures ----------------------------------------------------

WEEKDAYS = range(0, 5)  # Monday to Friday


@pytest.fixture
def calendar_settings():
    """Deterministic calendar configuration: UTC, on a 15-minute grid."""
    from app.config import Settings

    return Settings(
        _env_file=None, business_timezone="UTC", slot_granularity_minutes=15
    )


@pytest.fixture
def calendar(session, calendar_settings):
    from app.calendar import CalendarService

    return CalendarService(session, calendar_settings)


@pytest.fixture
def open_weekdays(session):
    """09:00–17:00, Monday to Friday."""
    from datetime import time

    from app.models import BusinessHours

    session.add_all(
        [
            BusinessHours(weekday=weekday, opens_at=time(9), closes_at=time(17))
            for weekday in WEEKDAYS
        ]
    )
    session.commit()


@pytest.fixture
def haircut(session):
    """A 30-minute service performed by staff member "sam"."""
    from app.models import Service

    service = Service(name="Haircut", duration_minutes=30, staff_id="sam")
    session.add(service)
    session.commit()
    return service


# --- tool fixtures --------------------------------------------------------


@pytest.fixture
def tools(session, calendar_settings):
    """A tool context on the test database, with no call attached."""
    from app.tools import ToolContext

    return ToolContext(session=session, settings=calendar_settings)


# --- dialogue fixtures ----------------------------------------------------


class FakeModel:
    """A `LanguageModel` that answers from a script, with no network.

    Records what it was asked so a test can assert the dialogue layer handed
    it the right system prompt, history and tools.
    """

    model_name = "fake-model-1"

    def __init__(self, *responses, raises: Exception | None = None) -> None:
        self._responses = list(responses)
        self._raises = raises
        self.requests: list[dict] = []

    def respond(self, *, system, messages, tools):
        self.requests.append(
            {"system": system, "messages": list(messages), "tools": list(tools)}
        )
        if self._raises is not None:
            raise self._raises
        if not self._responses:
            raise AssertionError("FakeModel ran out of scripted responses")
        return self._responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.requests)


def say(text: str, *, latency_ms: int = 12):
    """A scripted plain-text response."""
    from app.providers.llm import ModelResponse

    return ModelResponse(
        text=text,
        stop_reason="end_turn",
        raw_content=[{"type": "text", "text": text}],
        model_name="fake-model-1",
        latency_ms=latency_ms,
    )


def use_tools(*calls, text: str = "", latency_ms: int = 12):
    """A scripted response asking for one or more tools.

    Each call is `(name, arguments)`; ids are generated so they are unique
    within the response, as the API guarantees.
    """
    import uuid as _uuid

    from app.providers.llm import ModelResponse, ToolUse

    uses = [
        ToolUse(id=f"toolu_{_uuid.uuid4().hex[:12]}", name=name, arguments=arguments)
        for name, arguments in calls
    ]
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(
        {"type": "tool_use", "id": use.id, "name": use.name, "input": use.arguments}
        for use in uses
    )
    return ModelResponse(
        text=text,
        tool_uses=uses,
        stop_reason="tool_use",
        raw_content=content,
        model_name="fake-model-1",
        latency_ms=latency_ms,
    )


@pytest.fixture
def call(session):
    """A `Call` row for the dialogue layer to record turns against."""
    from app.models import Call, CallDirection

    row = Call(
        direction=CallDirection.INBOUND,
        from_number="+447700900123",
        to_number="+441234567890",
    )
    session.add(row)
    session.commit()
    return row


@pytest.fixture
def dialogue(session, call, calendar_settings):
    """Build a `Conversation` on the test database from a scripted model."""
    from app.dialogue import Conversation

    def build(*responses, raises=None, model=None, settings=None):
        return Conversation(
            session,
            call,
            model or FakeModel(*responses, raises=raises),
            settings or calendar_settings,
        )

    return build


# --- audio fixtures -------------------------------------------------------


def wav_bytes(
    duration_ms: int = 500,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """A silent WAV, deterministic and as wrong as a test asks it to be."""
    import io
    import wave

    frames = round(sample_rate * duration_ms / 1000)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00" * frames * channels * sample_width)
    return buffer.getvalue()


class FakeSTT:
    """A transcriber that answers from a script, with no network.

    Records the audio it was handed so a test can assert the audio layer
    passed the real bytes through rather than something it made up.
    """

    def __init__(self, *transcripts, raises: Exception | None = None) -> None:
        from app.providers.stt import Transcript

        self._scripted = [
            item if isinstance(item, Transcript) else Transcript(text=item)
            for item in (transcripts or ["I'd like to book an appointment"])
        ]
        self._raises = raises
        self.calls: list = []

    def transcribe(self, audio):
        self.calls.append(audio)
        if self._raises is not None:
            raise self._raises
        index = min(len(self.calls) - 1, len(self._scripted) - 1)
        return self._scripted[index]


class FakeTTS:
    """A synthesiser that returns fixed bytes, with no network."""

    def __init__(self, raises: Exception | None = None, sample_rate: int = 16000):
        from app.providers.speech import PCM_S16LE, AudioFormat

        self._format = AudioFormat(PCM_S16LE, sample_rate, 1, "wav")
        self._raises = raises
        self.calls: list[str] = []

    @property
    def format(self):
        return self._format

    def synthesize(self, text: str, voice: str | None = None):
        from app.providers.tts import Speech

        self.calls.append(text)
        if self._raises is not None:
            raise self._raises
        return Speech(
            audio=wav_bytes(200),
            format=self._format,
            duration_ms=200,
            voice=voice or "fake",
            latency_ms=1,
            provider_name="fake",
            characters=len(text),
        )


@pytest.fixture
def utterance():
    """One valid milestone-5 utterance."""
    from app.providers.speech import PCM_S16LE, Audio, AudioFormat

    data = wav_bytes(500)
    return Audio(
        data=data,
        format=AudioFormat(PCM_S16LE, 16000, 1, "wav"),
        duration_ms=500,
    )


@pytest.fixture
def voice(session, call, calendar_settings):
    """Build a `VoiceSession` over a scripted model, STT and TTS."""
    from app.audio import VoiceSession
    from app.dialogue import Conversation

    def build(*responses, stt=None, tts=None, model=None, raises=None, settings=None):
        resolved = settings or calendar_settings
        return VoiceSession(
            conversation=Conversation(
                session,
                call,
                model or FakeModel(*responses, raises=raises),
                resolved,
            ),
            stt=stt or FakeSTT(),
            tts=tts or FakeTTS(),
            settings=resolved,
        )

    return build


# --- telephony fixtures ---------------------------------------------------

TWILIO_STREAM_SID = "MZ00000000000000000000000000000001"
TWILIO_CALL_SID = "CA00000000000000000000000000000001"
TWILIO_AUTH_TOKEN = "test-auth-token"


def mulaw_frame(level: int = 0, duration_ms: int = 20) -> bytes:
    """One carrier frame of µ-law at a given loudness.

    `level` is a 16-bit PCM amplitude; 0 is silence and anything comfortably
    above the configured threshold counts as somebody speaking.
    """
    import struct

    from app.audio.telephony import mulaw_encode

    samples = int(8000 * duration_ms / 1000)
    pcm = struct.pack(
        f"<{samples}h", *[level if index % 2 else -level for index in range(samples)]
    )
    return mulaw_encode(pcm)


def twilio_frame(event: str, **body) -> str:
    """One Twilio Media Stream control frame, as JSON text."""
    import json

    return json.dumps({"event": event, **body})


def start_frame(
    call_sid: str = TWILIO_CALL_SID,
    stream_sid: str = TWILIO_STREAM_SID,
    **media_format,
) -> str:
    fields = {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
    fields.update(media_format)
    return twilio_frame(
        "start",
        streamSid=stream_sid,
        start={
            "callSid": call_sid,
            "accountSid": "AC1",
            "tracks": ["inbound"],
            "mediaFormat": fields,
        },
    )


def media_text_frame(
    audio: bytes, stream_sid: str = TWILIO_STREAM_SID, track: str = "inbound"
) -> str:
    import base64

    return twilio_frame(
        "media",
        streamSid=stream_sid,
        media={"track": track, "payload": base64.b64encode(audio).decode()},
    )


@pytest.fixture
def telephony_settings(calendar_settings):
    """Telephony switched on, signature checking off, offline providers."""
    return calendar_settings.model_copy(
        update={
            "telephony_enabled": True,
            "validate_twilio_signature": False,
            "twilio_auth_token": TWILIO_AUTH_TOKEN,
            "public_base_url": "https://voicedesk.example.com",
        }
    )


@pytest.fixture
def twilio_call(session):
    """A `Call` row of the kind the signed webhook creates."""
    from app.models import Call, CallDirection

    call = Call(
        direction=CallDirection.INBOUND,
        from_number="+447700900123",
        to_number="+441234567890",
        provider_call_sid=TWILIO_CALL_SID,
    )
    session.add(call)
    session.commit()
    return call


# --- realtime fixtures ----------------------------------------------------


def pcm_tone(
    amplitude: int = 9000,
    *,
    ms: int = 20,
    hz: float = 220.0,
    sample_rate: int = 8000,
) -> bytes:
    """One frame of a tone, loud enough to read as speech."""
    import math
    import struct

    count = int(sample_rate * ms / 1000)
    return struct.pack(
        f"<{count}h",
        *(int(amplitude * math.sin(2 * math.pi * hz * i / sample_rate)) for i in range(count)),
    )


def pcm_silence(*, ms: int = 20, sample_rate: int = 8000) -> bytes:
    return b"\x00\x00" * int(sample_rate * ms / 1000)


class FakeClock:
    """A clock a test moves by hand, so latency assertions are exact."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms


class RecordingSink:
    """An `AudioSink` that keeps everything, so a test can look at it."""

    def __init__(self) -> None:
        self.chunks: list = []
        self.clears = 0
        self.marks: list[str] = []
        self.order: list[str] = []

    async def send(self, chunk) -> None:
        self.chunks.append(chunk)
        self.order.append("send")

    async def clear(self) -> None:
        self.chunks.clear()
        self.clears += 1
        self.order.append("clear")

    async def mark(self, name: str) -> None:
        self.marks.append(name)
        self.order.append("mark")


@pytest.fixture
def realtime_settings(calendar_settings):
    """Realtime on, with the approved defaults."""
    return calendar_settings.model_copy(update={"realtime_enabled": True})


@pytest.fixture
def realtime(session, call, realtime_settings):
    """Build a `RealtimeSession` over scripted everything."""
    from app.dialogue import Conversation
    from app.providers.offline_streaming import (
        OfflineStreamingSpeechToText,
        OfflineStreamingTextToSpeech,
    )
    from app.realtime import RealtimeSession

    def build(
        *responses,
        stt=None,
        tts=None,
        model=None,
        raises=None,
        sink=None,
        settings=None,
        clock=None,
        on_partial=None,
        sample_rate: int = 8000,
    ):
        resolved = settings or realtime_settings
        built_sink = sink if sink is not None else RecordingSink()
        built = RealtimeSession(
            conversation=Conversation(
                session, call, model or FakeModel(*responses, raises=raises), resolved
            ),
            stt=stt or OfflineStreamingSpeechToText(),
            tts=tts or OfflineStreamingTextToSpeech(sample_rate=sample_rate),
            sink=built_sink,
            settings=resolved,
            sample_rate=sample_rate,
            clock=clock,
            on_partial=on_partial,
        )
        return built, built_sink

    return build


@pytest.fixture
def anyio_backend():
    """Async tests run on asyncio only; trio is not a dependency here."""
    return "asyncio"


@pytest.fixture
def db_session(session):
    """The database session, under a name that does not shadow other uses.

    The realtime tests call their session objects `session`, so the ORM one
    needs a different name in those files.
    """
    return session


# --- cost fixtures --------------------------------------------------------

# Invented for arithmetic, not copied from anybody's price list. They are
# deliberately round and deliberately wrong: no vendor charges these, and the
# point of every assertion below is the calculation, not the number.
FICTIONAL_LLM_INPUT_USD_PER_MTOK = "1"
FICTIONAL_LLM_OUTPUT_USD_PER_MTOK = "4"
FICTIONAL_STT_USD_PER_MINUTE = "0.12"
FICTIONAL_TTS_USD_PER_MCHAR = "500"


@pytest.fixture
def cost_settings(calendar_settings):
    """Cost tracking on, and not one price configured."""
    return calendar_settings.model_copy(update={"cost_tracking_enabled": True})


@pytest.fixture
def priced_settings(cost_settings):
    """Cost tracking on, with the fictional prices above."""
    return cost_settings.model_copy(
        update={
            "llm_input_usd_per_mtok": FICTIONAL_LLM_INPUT_USD_PER_MTOK,
            "llm_output_usd_per_mtok": FICTIONAL_LLM_OUTPUT_USD_PER_MTOK,
            "stt_usd_per_minute": FICTIONAL_STT_USD_PER_MINUTE,
            "tts_usd_per_mchar": FICTIONAL_TTS_USD_PER_MCHAR,
        }
    )


def spoke(
    text: str = "I'd like to book an appointment",
    *,
    audio_ms: int | None = 1500,
    provider_name: str = "offline",
):
    """A `Transcript` that reports how much audio it was given."""
    from app.providers.stt import Transcript

    return Transcript(
        text=text,
        confidence=0.9,
        audio_ms=audio_ms,
        latency_ms=5,
        provider_name=provider_name,
    )


def used(input_tokens: int | None, output_tokens: int | None, *, text: str = "Sure."):
    """A scripted plain-text response that reports token usage."""
    from app.providers.llm import ModelResponse

    return ModelResponse(
        text=text,
        stop_reason="end_turn",
        raw_content=[{"type": "text", "text": text}],
        model_name="fake-model-1",
        latency_ms=12,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def cost_rows(session, call_id):
    """Every cost row for one call, oldest first, as a component-keyed dict."""
    from sqlalchemy import select

    from app.models import CallCost

    rows = (
        session.execute(
            select(CallCost)
            .where(CallCost.call_id == call_id)
            .order_by(CallCost.created_at)
        )
        .scalars()
        .all()
    )
    return {row.component: row for row in rows}
