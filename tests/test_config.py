"""Environment-driven configuration."""

import pytest
from pydantic import ValidationError

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
    """Telephony and speech configuration belong to later milestones.

    The model settings arrived with milestone 4, which is what the dialogue
    layer needs; audio and telephony still have nothing to configure.
    """
    fields = set(Settings.model_fields)

    assert not {
        field
        for field in fields
        if any(
            token in field
            for token in ("twilio", "stt", "tts", "calendar", "sms", "email")
        )
    }


def test_the_dialogue_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key == ""
    assert settings.dialogue_model == "claude-opus-5"
    assert settings.dialogue_effort == "low"
    assert settings.max_tool_iterations == 8
    assert settings.dialogue_max_tokens == 1024


def test_an_unknown_effort_level_is_refused() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, dialogue_effort="enormous")


@pytest.mark.parametrize("value", [0, -1])
def test_a_loop_limit_that_cannot_run_is_refused(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_tool_iterations=value)
