"""Application configuration.

All settings come from the environment (or a local `.env`), never from
module-level constants scattered through the code. See `.env.example`.

Milestone 1 needs the application's identity and its database; milestone 4
adds the dialogue model. Telephony and speech configuration belong to the
milestones that introduce them and are deliberately absent.
"""

from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# What `output_config.effort` accepts, shallowest first.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


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
