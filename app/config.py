"""Application configuration.

All settings come from the environment (or a local `.env`), never from
module-level constants scattered through the code. See `.env.example`.

Milestone 1 needs the application's identity and its database; milestone 4
adds the dialogue model, milestone 5 the speech providers and milestone 6 the
telephony carrier. Messaging configuration belongs to the milestone that
introduces it and is deliberately absent.
"""

from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# What `output_config.effort` accepts, shallowest first.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Speech providers that can be selected by name. "offline" needs no
# credentials and no network, which is why it is the default: a fresh clone
# runs the whole browser harness without an account anywhere.
STT_PROVIDERS = ("offline", "deepgram")
TTS_PROVIDERS = ("offline", "elevenlabs")


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

    database_url: str = (
        "postgresql+psycopg://voicedesk:voicedesk@localhost:5432/voicedesk"
    )

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
