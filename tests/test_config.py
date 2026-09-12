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
    """Messaging and evaluation configuration belong to later milestones.

    The model settings arrived with milestone 4, the speech providers with
    milestone 5, the carrier with milestone 6 and cost tracking with milestone
    8. Nothing yet configures SMS, email or the evaluation suite.
    """
    fields = set(Settings.model_fields)

    assert not {
        field
        for field in fields
        if any(
            token in field
            for token in ("calendar", "sms", "email", "eval", "outbound")
        )
    }


def test_the_only_cost_settings_are_the_approved_ones() -> None:
    """The guard above used to forbid the word "cost" outright.

    It no longer can, so it is replaced by an exact list: cost tracking is one
    switch and four prices, and nothing has crept in beside them.
    """
    named = {
        field
        for field in Settings.model_fields
        if "cost" in field or "usd" in field
    }

    assert named == {
        "cost_tracking_enabled",
        "llm_input_usd_per_mtok",
        "llm_output_usd_per_mtok",
        "stt_usd_per_minute",
        "tts_usd_per_mchar",
    }


def test_cost_tracking_is_off_and_every_price_is_unset() -> None:
    """A fresh clone records no cost rows and asserts no price."""
    settings = Settings(_env_file=None)

    assert settings.cost_tracking_enabled is False
    assert settings.llm_input_usd_per_mtok == ""
    assert settings.llm_output_usd_per_mtok == ""
    assert settings.stt_usd_per_minute == ""
    assert settings.tts_usd_per_mchar == ""


def test_no_setting_configures_a_telephony_price() -> None:
    """Measured stream time is not what a carrier bills, so it is not priced."""
    assert not {
        field
        for field in Settings.model_fields
        if "telephony" in field and ("usd" in field or "price" in field)
    }


def test_the_telephony_settings_have_safe_defaults() -> None:
    """Nothing telephonic is reachable until somebody turns it on."""
    settings = Settings(_env_file=None)

    assert settings.telephony_enabled is False
    assert settings.validate_twilio_signature is True
    assert settings.twilio_account_sid == ""
    assert settings.twilio_auth_token == ""
    assert settings.twilio_phone_number == ""
    assert settings.public_base_url == ""


def test_the_utterance_boundary_has_the_approved_defaults() -> None:
    """Temporary milestone-6 mechanics; the real-time milestone replaces them."""
    settings = Settings(_env_file=None)

    assert settings.telephony_silence_ms == 800
    assert settings.telephony_silence_threshold == 500
    assert settings.telephony_max_utterance_ms == 30_000


@pytest.mark.parametrize(
    "field", ["telephony_silence_ms", "telephony_max_utterance_ms"]
)
def test_an_utterance_boundary_that_cannot_work_is_refused(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 0})


def test_a_negative_silence_threshold_is_refused() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, telephony_silence_threshold=-1)


def test_the_speech_settings_have_safe_defaults() -> None:
    """A fresh clone runs the whole harness with no account anywhere."""
    settings = Settings(_env_file=None)

    assert settings.stt_provider == "offline"
    assert settings.tts_provider == "offline"
    assert settings.deepgram_api_key == ""
    assert settings.elevenlabs_api_key == ""
    assert settings.audio_sample_rate == 16000
    assert settings.max_utterance_bytes == 1_048_576
    assert settings.speech_timeout_seconds == 10.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("stt_provider", "whisper"), ("tts_provider", "polly")],
)
def test_an_unknown_speech_provider_is_refused(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize("field", ["audio_sample_rate", "max_utterance_bytes"])
def test_an_audio_setting_that_cannot_work_is_refused(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 0})


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
