"""Application configuration.

All settings come from the environment (or a local `.env`), never from
module-level constants scattered through the code. See `.env.example`.

Milestone 1 needs the application's identity and its database. Telephony,
speech and model configuration belong to the milestones that introduce them
and are deliberately absent.
"""

from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
