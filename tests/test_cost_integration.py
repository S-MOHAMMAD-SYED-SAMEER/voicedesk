"""Cost tracking through the whole stack, against a real database.

Nothing here reaches a vendor. The model, the transcriber and the synthesiser
are scripted, and the prices are the invented ones from `conftest.py`.

The test this file exists for is
`test_a_reply_split_into_many_chunks_is_counted_once`: every streaming
synthesiser in this repository repeats the full character count on every chunk
it emits, so a layer that summed them would multiply the bill by the number of
chunks. That has to be provably not happening.
"""

import base64
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.audio.telephony import mulaw_encode
from app.models import Call, CostComponent
from app.providers.offline_streaming import (
    OfflineStreamingSpeechToText,
    OfflineStreamingTextToSpeech,
)
from app.providers.speech import PCM_S16LE, AudioFormat
from app.providers.streaming_tts import SpeechChunk

from .conftest import (
    TWILIO_STREAM_SID,
    FakeModel,
    FakeSTT,
    FakeTTS,
    cost_rows,
    pcm_silence,
    pcm_tone,
    say,
    spoke,
    start_frame,
    twilio_frame,
    use_tools,
    used,
    wav_bytes,
)

LOUD_16K = pcm_tone(9000, sample_rate=16000)
QUIET_16K = pcm_silence(sample_rate=16000)
LOUD_8K = mulaw_encode(pcm_tone(9000))
QUIET_8K = mulaw_encode(pcm_silence())

STREAM = "/telephony/stream"


# --- harnesses ------------------------------------------------------------


@pytest.fixture
def harness(migrated_engine: Engine, priced_settings, monkeypatch):
    """The browser harness, with scripted providers and invented prices."""
    from app.main import create_app

    def build(*responses, model=None, stt=None, tts=None, **overrides):
        settings = priced_settings.model_copy(update=overrides)
        scripted = model or FakeModel(*responses)
        monkeypatch.setattr("app.api.harness.get_settings", lambda: settings)
        monkeypatch.setattr(
            "app.api.harness.build_model", lambda settings=None: scripted
        )
        monkeypatch.setattr(
            "app.api.harness.build_stt",
            lambda settings=None: stt or FakeSTT(spoke()),
        )
        monkeypatch.setattr(
            "app.api.harness.build_tts", lambda settings=None: FakeTTS()
        )
        monkeypatch.setattr(
            "app.api.harness.build_streaming_stt",
            lambda settings=None: OfflineStreamingSpeechToText(),
        )
        monkeypatch.setattr(
            "app.api.harness.build_streaming_tts",
            lambda settings=None: tts
            or OfflineStreamingTextToSpeech(
                sample_rate=16000, chunk_ms=200, ms_per_character=2
            ),
        )
        return TestClient(create_app()), scripted

    return build


@pytest.fixture
def phone(migrated_engine: Engine, telephony_settings, priced_settings, monkeypatch):
    """A telephony stream on the realtime path, with cost tracking on."""
    from app.main import create_app

    def build(*responses, **overrides):
        fields = {
            "realtime_enabled": True,
            "cost_tracking_enabled": True,
            "llm_input_usd_per_mtok": priced_settings.llm_input_usd_per_mtok,
            "llm_output_usd_per_mtok": priced_settings.llm_output_usd_per_mtok,
            "stt_usd_per_minute": priced_settings.stt_usd_per_minute,
            "tts_usd_per_mchar": priced_settings.tts_usd_per_mchar,
        }
        fields.update(overrides)
        settings = telephony_settings.model_copy(update=fields)
        scripted = FakeModel(*responses)
        monkeypatch.setattr("app.telephony.stream.get_settings", lambda: settings)
        monkeypatch.setattr(
            "app.telephony.stream.build_model", lambda settings=None: scripted
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_streaming_stt",
            lambda settings=None: OfflineStreamingSpeechToText(),
        )
        monkeypatch.setattr(
            "app.telephony.stream.build_streaming_tts",
            lambda settings=None: OfflineStreamingTextToSpeech(
                sample_rate=8000, chunk_ms=200, ms_per_character=2
            ),
        )
        return TestClient(create_app()), scripted

    return build


def _ready(socket) -> dict:
    message = socket.receive_json()
    socket.receive_bytes()  # the greeting audio
    return message


def _drain(socket, limit: int = 400) -> list[dict]:
    seen: list[dict] = []
    for _ in range(limit):
        try:
            message = socket.receive_json()
        except Exception:
            break
        seen.append(message)
        if message.get("event") in {"mark", "clear"}:
            break
        if message.get("type") == "turn":
            break
    return seen


def _media(audio: bytes) -> str:
    return twilio_frame(
        "media",
        streamSid=TWILIO_STREAM_SID,
        media={"track": "inbound", "payload": base64.b64encode(audio).decode()},
    )


def _latest_call(session: Session) -> Call:
    return (
        session.execute(select(Call).order_by(Call.started_at.desc()))
        .scalars()
        .first()
    )


# --- the trap: characters counted once per stream -------------------------


class ManyChunks:
    """A synthesiser that splits a reply into ten chunks.

    Every chunk reports the *whole* text length, which is what every real
    streaming implementation in this repository does. A layer that summed them
    would record ten times the truth.
    """

    CHUNKS = 10

    def __init__(self, sample_rate: int = 16000) -> None:
        self._format = AudioFormat(PCM_S16LE, sample_rate, 1, "raw")

    def stream(self, text: str, voice: str | None = None):
        return _ManyChunkStream(text, self._format)


class _ManyChunkStream:
    def __init__(self, text: str, audio_format: AudioFormat) -> None:
        self._text = text
        self._format = audio_format

    async def chunks(self):
        for index in range(ManyChunks.CHUNKS):
            yield SpeechChunk(
                audio=b"\x00\x00" * 160,
                format=self._format,
                is_final=index == ManyChunks.CHUNKS - 1,
                # The whole count, on every chunk. This is the trap.
                characters=len(self._text),
                # A vendor, so the invented price applies and the arithmetic
                # is visible. An offline provider would cost zero and prove
                # nothing about summing.
                metadata={"provider": "elevenlabs"},
            )

    async def aclose(self) -> None:
        return None


REPLY = "x" * 100


def test_a_reply_split_into_many_chunks_is_counted_once(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    """A hundred characters in ten chunks is a hundred characters."""
    client, _ = harness(
        used(50, 10, text=REPLY), tts=ManyChunks(), realtime_enabled=True
    )

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        for _ in range(20):
            socket.send_bytes(LOUD_16K)
        for _ in range(40):
            socket.send_bytes(QUIET_16K)
        _drain(socket)

    call = _latest_call(db_session)
    row = cost_rows(db_session, call.id)[CostComponent.TTS]
    assert row.input_units == Decimal("100.000")
    assert row.input_units != Decimal(100 * ManyChunks.CHUNKS)


def test_the_cost_of_that_reply_is_one_reply_not_ten(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(
        used(50, 10, text=REPLY), tts=ManyChunks(), realtime_enabled=True
    )

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        for _ in range(20):
            socket.send_bytes(LOUD_16K)
        for _ in range(40):
            socket.send_bytes(QUIET_16K)
        _drain(socket)

    call = _latest_call(db_session)
    # 100 characters at the invented 0.0005 each.
    assert cost_rows(db_session, call.id)[CostComponent.TTS].cost_usd == Decimal(
        "0.050000"
    )


# --- push-to-talk ---------------------------------------------------------


def test_a_press_to_talk_turn_records_every_component(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(used(1000, 200))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    assert set(cost_rows(db_session, call.id)) == {
        CostComponent.LLM,
        CostComponent.STT,
        CostComponent.TTS,
    }


def test_a_press_to_talk_turn_records_the_audio_it_was_given(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(used(1000, 200))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    row = cost_rows(db_session, call.id)[CostComponent.STT]
    assert row.input_units == Decimal("1500.000")
    assert row.provider == "offline"


def test_a_browser_call_has_no_line_row(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    """There is no carrier, so there is nothing to record about one."""
    client, _ = harness(used(1000, 200))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    assert CostComponent.TELEPHONY not in cost_rows(db_session, call.id)


def test_a_fully_priced_browser_call_gets_a_total(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(used(1000, 200))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    db_session.refresh(call)
    assert call.total_cost_usd is not None
    assert call.total_cost_usd > Decimal("0")


def test_an_unpriced_component_leaves_the_browser_call_without_a_total(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(
        used(1000, 200),
        llm_input_usd_per_mtok="",
        llm_output_usd_per_mtok="",
    )

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    db_session.refresh(call)
    rows = cost_rows(db_session, call.id)
    assert rows[CostComponent.LLM].cost_usd is None
    assert call.total_cost_usd is None


def test_a_model_that_reports_no_tokens_writes_no_model_row(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    """`say()` reports no usage, which is not the same as reporting zero."""
    client, _ = harness(say("Certainly."))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    assert CostComponent.LLM not in cost_rows(db_session, call.id)


def test_with_cost_tracking_off_nothing_is_recorded(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(used(1000, 200), cost_tracking_enabled=False)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    db_session.refresh(call)
    assert cost_rows(db_session, call.id) == {}
    assert call.total_cost_usd is None


def test_a_turn_the_model_never_saw_records_nothing_and_under_counts(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    """A known, asserted limitation rather than a quiet one.

    When the caller says nothing recognisable the model is deliberately not
    asked, so the dialogue layer writes no turn rows — and a cost row hangs
    off a turn. The recogniser and the synthesiser were still used and will
    still be billed, and that spend is not recorded here.

    Attaching it to the call instead would be worse: the partial index makes
    call-level rows unique per component, so a caller who was misheard three
    times would have two of those three silently overwritten. A visible
    under-count beats an invisible one.
    """
    client, _ = harness(used(1, 1), stt=FakeSTT(spoke("   ")))

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    assert cost_rows(db_session, call.id) == {}


# --- the tool loop --------------------------------------------------------


def test_tokens_are_summed_across_every_request_a_turn_made(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    """A turn is one reply but may be several requests, all of them paid for."""
    asking = use_tools(
        ("check_availability", {"service_name": "Haircut", "date": "2026-03-02"})
    )
    client, _ = harness(
        asking.__class__(
            text=asking.text,
            tool_uses=asking.tool_uses,
            stop_reason=asking.stop_reason,
            raw_content=asking.raw_content,
            model_name="fake-model-1",
            input_tokens=500,
            output_tokens=50,
            latency_ms=12,
        ),
        used(700, 80, text="We have 9am free."),
    )

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        socket.send_bytes(wav_bytes(500))
        socket.receive_json()
        socket.receive_bytes()

    call = _latest_call(db_session)
    row = cost_rows(db_session, call.id)[CostComponent.LLM]
    assert row.input_units == Decimal("1200.000")
    assert row.output_units == Decimal("130.000")


# --- realtime -------------------------------------------------------------


def test_a_realtime_turn_records_every_component(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(used(1000, 200), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        for _ in range(20):
            socket.send_bytes(LOUD_16K)
        for _ in range(40):
            socket.send_bytes(QUIET_16K)
        _drain(socket)

    call = _latest_call(db_session)
    assert set(cost_rows(db_session, call.id)) == {
        CostComponent.LLM,
        CostComponent.STT,
        CostComponent.TTS,
    }


def test_a_realtime_turn_names_the_offline_providers(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(used(1000, 200), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        for _ in range(20):
            socket.send_bytes(LOUD_16K)
        for _ in range(40):
            socket.send_bytes(QUIET_16K)
        _drain(socket)

    call = _latest_call(db_session)
    rows = cost_rows(db_session, call.id)
    assert rows[CostComponent.STT].provider == "offline"
    assert rows[CostComponent.TTS].provider == "offline"


def test_the_offline_providers_are_free_and_the_model_is_not(
    db_session: Session, harness, open_weekdays, haircut
) -> None:
    client, _ = harness(used(1000, 200), realtime_enabled=True)

    with client.websocket_connect("/ws/harness") as socket:
        _ready(socket)
        for _ in range(20):
            socket.send_bytes(LOUD_16K)
        for _ in range(40):
            socket.send_bytes(QUIET_16K)
        _drain(socket)

    call = _latest_call(db_session)
    rows = cost_rows(db_session, call.id)
    assert rows[CostComponent.STT].cost_usd == Decimal("0.000000")
    assert rows[CostComponent.TTS].cost_usd == Decimal("0.000000")
    assert rows[CostComponent.LLM].cost_usd == Decimal("0.001800")


# --- a telephone call -----------------------------------------------------


def test_a_phone_call_records_how_long_the_line_was_open(
    db_session: Session, phone, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = phone(used(1000, 200))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    db_session.expire_all()
    rows = cost_rows(db_session, twilio_call.id)
    assert CostComponent.TELEPHONY in rows
    assert rows[CostComponent.TELEPHONY].unit_type == "duration_ms"
    assert rows[CostComponent.TELEPHONY].provider == "twilio"


def test_the_line_on_a_phone_call_is_recorded_unpriced(
    db_session: Session, phone, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = phone(used(1000, 200))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    db_session.expire_all()
    rows = cost_rows(db_session, twilio_call.id)
    assert rows[CostComponent.TELEPHONY].cost_usd is None


def test_a_phone_call_therefore_has_no_total(
    db_session: Session, phone, twilio_call, open_weekdays, haircut
) -> None:
    """Stated as a test so nobody discovers it as a surprise."""
    client, _ = phone(used(1000, 200))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    db_session.expire_all()
    db_session.refresh(twilio_call)
    assert twilio_call.total_cost_usd is None


def test_a_phone_turn_records_the_model_and_the_speech(
    db_session: Session, phone, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = phone(used(1000, 200))

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain(socket)
        for _ in range(10):
            socket.send_text(_media(LOUD_8K))
        for _ in range(40):
            socket.send_text(_media(QUIET_8K))
        _drain(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    db_session.expire_all()
    rows = cost_rows(db_session, twilio_call.id)
    assert rows[CostComponent.LLM].input_units == Decimal("1000.000")
    assert rows[CostComponent.TTS].input_units > Decimal("0")


def test_a_phone_call_with_cost_tracking_off_records_nothing(
    db_session: Session, phone, twilio_call, open_weekdays, haircut
) -> None:
    client, _ = phone(used(1000, 200), cost_tracking_enabled=False)

    with client.websocket_connect(STREAM) as socket:
        socket.send_text(start_frame())
        _drain(socket)
        socket.send_text(twilio_frame("stop", streamSid=TWILIO_STREAM_SID))

    db_session.expire_all()
    assert cost_rows(db_session, twilio_call.id) == {}


# --- the application refuses to start on a bad price ----------------------


def test_a_malformed_price_stops_the_application_starting(
    migrated_engine: Engine, cost_settings, monkeypatch
) -> None:
    """Discovered at boot, not one swallowed exception at a time."""
    from app.cost import PricingError
    from app.main import create_app

    settings = cost_settings.model_copy(update={"tts_usd_per_mchar": "free"})
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    with pytest.raises(PricingError, match="VOICEDESK_TTS_USD_PER_MCHAR"):
        create_app()


def test_a_bad_price_does_not_stop_an_application_with_tracking_off(
    migrated_engine: Engine, calendar_settings, monkeypatch
) -> None:
    settings = calendar_settings.model_copy(update={"tts_usd_per_mchar": "free"})
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    from app.main import create_app

    assert create_app() is not None
