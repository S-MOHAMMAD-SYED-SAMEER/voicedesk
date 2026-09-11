"""Environment-driven configuration."""

import pytest

from app.config import Settings, get_settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_defaults() -> None:
    settings = _settings()

    assert settings.app_name == "VoiceDesk"
    assert settings.environment == "local"
    assert settings.debug is False


def test_database_url_defaults_to_the_psycopg_driver() -> None:
    assert _settings().database_url.startswith("postgresql+psycopg://")


def test_the_default_database_is_not_docintels() -> None:
    """VoiceDesk must never be pointed at the other project's database."""
    assert "docintel" not in _settings().database_url


@pytest.mark.parametrize(
    ("variable", "attribute", "value", "expected"),
    [
        ("VOICEDESK_ENVIRONMENT", "environment", "staging", "staging"),
        ("VOICEDESK_DEBUG", "debug", "true", True),
        (
            "VOICEDESK_DATABASE_URL",
            "database_url",
            "postgresql+psycopg://a:b@db.example:5432/c",
            "postgresql+psycopg://a:b@db.example:5432/c",
        ),
    ],
)
def test_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    attribute: str,
    value: str,
    expected: object,
) -> None:
    monkeypatch.setenv(variable, value)
    get_settings.cache_clear()
    try:
        assert getattr(get_settings(), attribute) == expected
    finally:
        get_settings.cache_clear()


def test_the_prefix_is_voicedesk(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unprefixed variable must not leak in from another project."""
    monkeypatch.setenv("ENVIRONMENT", "leaked")
    get_settings.cache_clear()
    try:
        assert get_settings().environment != "leaked"
    finally:
        get_settings.cache_clear()


def test_settings_are_cached(settings_env: None) -> None:
    assert get_settings() is get_settings()


def test_no_future_milestone_settings_exist() -> None:
    """Telephony, speech and model configuration belong to later milestones."""
    fields = set(Settings.model_fields)

    assert not {
        field
        for field in fields
        if any(
            token in field
            for token in ("twilio", "anthropic", "stt", "tts", "calendar", "sms")
        )
    }
