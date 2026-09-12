"""Application configuration.

All settings come from the environment (or a local `.env`), never from
module-level constants scattered through the code. See `.env.example`.

Milestone 1 needs the application's identity and its database; milestone 4
adds the dialogue model, milestone 5 the speech providers, milestone 6 the
telephony carrier and milestone 7 realtime voice. Messaging configuration
belongs to the milestone that introduces it and is deliberately absent.
"""

from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# What `output_config.effort` accepts, shallowest first.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# The environment name that turns on every production guard at once: the
# preflight in `app/preflight.py`, the closed documentation surface, and the
# refusal to serve the development harness.
PRODUCTION = "production"

# Local development only. Named here so the preflight can recognise it and
# refuse to let it become a production connection target.
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://voicedesk:voicedesk@localhost:5432/voicedesk"
)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMATS = ("text", "json")

# Speech providers that can be selected by name. "offline" needs no
# credentials and no network, which is why it is the default: a fresh clone
# runs the whole browser harness without an account anywhere.
STT_PROVIDERS = ("offline", "deepgram")
TTS_PROVIDERS = ("offline", "elevenlabs")
# The streaming counterparts, selected separately: a deployment may want
# one vendor for whole utterances and another for realtime, and the two
# interfaces are genuinely different protocols.
STREAMING_STT_PROVIDERS = ("offline", "deepgram")
STREAMING_TTS_PROVIDERS = ("offline", "elevenlabs")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOICEDESK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "VoiceDesk"
    environment: str = "local"
    debug: bool = False

    # A local development default, and deliberately not a usable production
    # one. `app/preflight.py` refuses to start in production while this is
    # still what `database_url` says, so a deployment that forgets to
    # configure a database fails at boot rather than quietly connecting to
    # localhost with a well-known password.
    database_url: str = DEFAULT_DATABASE_URL

    # --- Calendar ---
    # `business_hours` stores wall-clock times and appointments are instants,
    # so something has to say which wall clock. The specification defines no
    # timezone and VoiceDesk serves one business (multi-tenancy is a non-goal),
    # so it is one setting rather than a column. UTC by default; a real
    # deployment sets its own.
    business_timezone: str = "UTC"
    # The grid available start times sit on, measured from each opening time.
    # The specification does not name a granularity; 15 minutes is the usual
    # booking grid and is deterministic.
    slot_granularity_minutes: int = Field(default=15, gt=0)

    # --- Dialogue ---
    # Empty by default so nothing in the repository implies a credential. The
    # SDK falls back to its own environment resolution when this is blank, and
    # no test ever needs a key: the model is injected.
    anthropic_api_key: str = ""
    dialogue_model: str = "claude-opus-5"
    # A receptionist answers in about two sentences. The ceiling is for a turn
    # that also carries tool calls.
    dialogue_max_tokens: int = Field(default=1024, gt=0)
    # Thinking depth. The specification's latency budget is 1.2 seconds from
    # the caller stopping to the agent speaking, so the default is the
    # shallowest setting rather than the API's own default of `high`.
    dialogue_effort: str = "low"
    # How many times one caller turn may go round the model/tool loop before
    # the dialogue layer stops it. Reaching this is a failure, not an answer.
    max_tool_iterations: int = Field(default=8, gt=0)

    # --- Speech ---
    # Offline by default and every key blank, so nothing in the repository
    # implies a credential and `pip install` to a working harness needs no
    # account. The offline providers are a fixed transcript and a tone; they
    # are a harness, not speech recognition or a voice.
    stt_provider: str = "offline"
    tts_provider: str = "offline"
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""
    stt_model: str = "nova-3"
    tts_voice: str = ""
    tts_model: str = ""
    # The one audio format milestone 5 accepts, in Hz. Telephony is 8 kHz
    # µ-law and arrives with the milestone that needs it.
    audio_sample_rate: int = Field(default=16000, gt=0)
    # One utterance. At 16 kHz mono 16-bit this is roughly 32 seconds, which
    # is far more than a caller says in one breath and small enough that a
    # malformed or hostile frame costs nothing.
    max_utterance_bytes: int = Field(default=1_048_576, gt=0)
    speech_timeout_seconds: float = Field(default=10.0, gt=0)

    # --- Telephony ---
    # Off, and every credential blank. Nothing telephonic is reachable until
    # somebody deliberately turns it on, and a fresh clone still runs the
    # browser harness and the whole test suite without an account anywhere.
    telephony_enabled: bool = False
    twilio_account_sid: str = ""
    # Also the key the webhook signature is checked against. A secret: never
    # logged, never echoed in an error.
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    # The address the carrier dials back for the media stream. It cannot be
    # derived from the request: behind a proxy the request describes the hop,
    # not the address reachable from the internet.
    public_base_url: str = ""
    # Off only for replaying captured requests locally. An unsigned webhook is
    # an open door to this database and this account's model budget.
    validate_twilio_signature: bool = True

    # --- The milestone-6 utterance boundary ---
    # Telephony streams continuously; the dialogue layer wants complete
    # utterances. This is an amplitude timer and nothing more: no spectral
    # analysis, no adaptive noise floor, no speech classifier. It is the
    # minimum mechanics without which a phone call cannot produce a turn, and
    # the milestone that owns real-time behaviour replaces it entirely.
    telephony_silence_ms: int = Field(default=800, gt=0)
    telephony_silence_threshold: int = Field(default=500, ge=0)
    telephony_max_utterance_ms: int = Field(default=30_000, gt=0)

    # --- Realtime ---
    # Off by default, so the milestone-6 behaviour — a whole utterance, then a
    # whole reply — is exactly what runs until somebody opts in.
    realtime_enabled: bool = False
    stt_streaming_provider: str = "offline"
    tts_streaming_provider: str = "offline"
    # What a carrier sends, and therefore the grain everything works in.
    audio_frame_ms: int = Field(default=20, gt=0)

    # --- Voice activity ---
    # An energy detector, not a trained one: how much louder than the room
    # something has to be before it counts as somebody talking. See
    # `app/audio/vad.py` for what that does and does not buy.
    vad_min_speech_ms: int = Field(default=120, gt=0)
    vad_energy_threshold: int = Field(default=300, ge=0)
    # How fast the ambient level is re-learned from quiet frames.
    vad_noise_floor_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    # Leaving speech is easier than entering it, so one quiet frame between
    # words does not end a sentence.
    vad_hysteresis: float = Field(default=0.6, gt=0.0, le=1.0)
    # Audio kept from before the detector was convinced, so the first syllable
    # is not clipped.
    vad_preroll_ms: int = Field(default=200, ge=0)

    # --- Endpointing ---
    endpoint_silence_ms: int = Field(default=700, gt=0)
    # A caller who never pauses is answered anyway. Also what bounds the
    # utterance buffer on an open socket.
    endpoint_max_utterance_ms: int = Field(default=20_000, gt=0)

    # --- Interruption ---
    barge_in_enabled: bool = True

    # --- Cost ---
    # Off by default: with it false nothing writes a cost row and
    # `calls.total_cost_usd` stays null, exactly as it has been since
    # milestone 1.
    cost_tracking_enabled: bool = False
    # Prices are strings, and empty means unpriced. This is the whole of
    # VoiceDesk's pricing: it ships no vendor prices and never infers one, so
    # a component nobody has priced is recorded with its usage and a null
    # cost. A numeric field defaulting to 0.0 would price every call at
    # nothing, which is the one answer that must never be given by accident.
    #
    # They are quoted in the units vendors quote in — per million tokens, per
    # minute, per million characters — and converted to per-unit rates in
    # `app/cost/pricing.py`. One price per component: if the model or the
    # speech provider changes, the price has to change with it, because
    # nothing here notices that it did not.
    llm_input_usd_per_mtok: str = ""
    llm_output_usd_per_mtok: str = ""
    stt_usd_per_minute: str = ""
    tts_usd_per_mchar: str = ""
    # There is deliberately no telephony price. What this system can measure
    # is how long a media stream was open; what a carrier bills is its own
    # record of the call, rounded up, which this process never sees.

    # --- Production ---
    # The browser harness is a development tool that spends model budget and
    # writes real rows, so production never serves it whatever this says —
    # see `Settings.harness_available`. The switch exists so a shared staging
    # deployment can turn it off too.
    harness_enabled: bool = True
    # How long a draining process waits for calls already in progress before
    # it stops. Long enough for a caller to finish a sentence and hear the
    # answer; short enough that a deployment is not held up by one open
    # socket.
    shutdown_grace_seconds: float = Field(default=20.0, gt=0)
    # Calls accepted at once. Each holds one database connection for its
    # whole length, and the default pool is 5 + 10 overflow, so 20 would
    # exhaust it — the cap is here to reject the twenty-first caller quickly
    # rather than let them block on a connection that is not coming.
    max_concurrent_calls: int = Field(default=10, gt=0)
    # A call that never ends. An hour is far longer than any receptionist
    # conversation and short enough to bound a forgotten socket.
    max_call_seconds: int = Field(default=3600, gt=0)
    # A voice webhook is a few hundred bytes of form data. 64 KiB is two
    # orders of magnitude of headroom and still refuses a body sent to
    # exhaust memory.
    max_webhook_bytes: int = Field(default=64 * 1024, gt=0)
    # One carrier control frame. Media frames are 20 ms of µ-law in base64,
    # well under a kilobyte; this bounds a frame sent to be large.
    max_stream_frame_bytes: int = Field(default=128 * 1024, gt=0)
    # How long the token in a stream URL stays usable. A carrier connects
    # within seconds of the webhook; five minutes is generous.
    stream_token_ttl_seconds: int = Field(default=300, gt=0)

    log_level: str = "INFO"
    log_format: str = "text"

    # --- Timeouts ---
    # Every outbound boundary has one. A caller is on the telephone: a
    # provider that has not answered in this long is a provider that has
    # failed, whatever it does afterwards.
    #
    # The model request runs on a worker thread and a timeout there cannot
    # kill that thread — it ends the *wait*, and the abandoned request
    # finishes into nothing. See the README.
    dialogue_timeout_seconds: float = Field(default=30.0, gt=0)
    stream_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    # Between messages, not for the whole stream: a synthesiser sending audio
    # steadily is working, however long the reply is.
    stream_read_timeout_seconds: float = Field(default=20.0, gt=0)
    database_connect_timeout_seconds: int = Field(default=10, gt=0)

    # --- Evaluation ---
    # Where `python -m app.evals` runs. Blank means "derive one from
    # `database_url` by appending `_evals`", which is the safe default: the
    # evaluation suite creates, migrates and truncates whatever it is pointed
    # at, so it must never be pointed at the database anything else uses. The
    # runner refuses outright to touch a database whose name does not end in
    # `_evals`, whether that name was derived or configured here.
    eval_database_url: str = ""

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() == PRODUCTION

    @property
    def harness_available(self) -> bool:
        """Whether the development harness may be served at all.

        Two conditions, and production overrides the switch: an operator who
        leaves `harness_enabled` at its default must still not end up with an
        unauthenticated page that spends model budget.
        """
        return self.harness_enabled and not self.is_production

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in LOG_LEVELS:
            raise ValueError(
                f"{value!r} is not a log level; use one of "
                f"{', '.join(LOG_LEVELS)}"
            )
        return level

    @field_validator("log_format")
    @classmethod
    def _known_log_format(cls, value: str) -> str:
        chosen = value.strip().lower()
        if chosen not in LOG_FORMATS:
            raise ValueError(
                f"{value!r} is not a log format; use one of "
                f"{', '.join(LOG_FORMATS)}"
            )
        return chosen

    @field_validator("stt_streaming_provider")
    @classmethod
    def _known_streaming_stt(cls, value: str) -> str:
        if value not in STREAMING_STT_PROVIDERS:
            raise ValueError(
                f"{value!r} is not a streaming speech-to-text provider; use one "
                f"of {', '.join(STREAMING_STT_PROVIDERS)}"
            )
        return value

    @field_validator("tts_streaming_provider")
    @classmethod
    def _known_streaming_tts(cls, value: str) -> str:
        if value not in STREAMING_TTS_PROVIDERS:
            raise ValueError(
                f"{value!r} is not a streaming text-to-speech provider; use one "
                f"of {', '.join(STREAMING_TTS_PROVIDERS)}"
            )
        return value

    @field_validator("stt_provider")
    @classmethod
    def _known_stt_provider(cls, value: str) -> str:
        if value not in STT_PROVIDERS:
            raise ValueError(
                f"{value!r} is not a speech-to-text provider; use one of "
                f"{', '.join(STT_PROVIDERS)}"
            )
        return value

    @field_validator("tts_provider")
    @classmethod
    def _known_tts_provider(cls, value: str) -> str:
        if value not in TTS_PROVIDERS:
            raise ValueError(
                f"{value!r} is not a text-to-speech provider; use one of "
                f"{', '.join(TTS_PROVIDERS)}"
            )
        return value

    @field_validator("dialogue_effort")
    @classmethod
    def _known_effort(cls, value: str) -> str:
        if value not in EFFORT_LEVELS:
            raise ValueError(
                f"{value!r} is not an effort level; use one of "
                f"{', '.join(EFFORT_LEVELS)}"
            )
        return value

    @field_validator("business_timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"{value!r} is not a known IANA timezone") from exc
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, safe to use as a FastAPI dependency."""
    return Settings()
