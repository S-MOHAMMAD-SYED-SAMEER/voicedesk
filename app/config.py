"""Application configuration.

All settings come from the environment (or a local `.env`), never from
module-level constants scattered through the code. See `.env.example`.

Milestone 1 needs the application's identity and its database. Telephony,
speech and model configuration belong to the milestones that introduce them
and are deliberately absent.
"""

from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, safe to use as a FastAPI dependency."""
    return Settings()
