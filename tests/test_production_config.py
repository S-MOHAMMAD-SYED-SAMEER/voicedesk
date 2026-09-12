"""What production refuses to start without, and what it must never require.

Two halves, and the second matters as much as the first. A preflight that
made the test suite need an API key, or the evaluation suite need a carrier
account, would have bought deployment safety by breaking the thing this
repository is built around: everything runs offline, with no credential
anywhere, by default.
"""

import pytest

from app.config import DEFAULT_DATABASE_URL, Settings
from app.preflight import ConfigurationError, check, problems

REAL_DB = "postgresql+psycopg://someone:secret@db.example:5432/voicedesk"


def _production(**overrides) -> Settings:
    fields = {
        "environment": "production",
        "database_url": REAL_DB,
        "anthropic_api_key": "sk-test-not-a-real-key",
    }
    fields.update(overrides)
    return Settings(_env_file=None, **fields)


def _settings(**overrides) -> str:
    return {problem.setting for problem in problems(_production(**overrides))}


# --- nothing is required outside production --------------------------------


def test_a_default_configuration_has_no_problems() -> None:
    """A fresh clone runs with no credential at all. That must not change."""
    assert problems(Settings(_env_file=None)) == []


def test_the_test_environment_has_no_problems() -> None:
    assert problems(Settings(_env_file=None, environment="test")) == []


def test_a_development_configuration_with_nothing_set_is_fine() -> None:
    """Offline providers, no keys, no carrier. This is the default posture."""
    settings = Settings(
        _env_file=None,
        environment="local",
        anthropic_api_key="",
        stt_provider="offline",
        tts_provider="offline",
    )

    assert problems(settings) == []
    check(settings)


def test_evaluation_settings_need_no_credential() -> None:
    """`python -m app.evals` must keep running with nothing configured."""
    settings = Settings(_env_file=None, environment="test", cost_tracking_enabled=True)

    assert problems(settings) == []


def test_telephony_off_needs_no_carrier_configuration() -> None:
    assert _settings(telephony_enabled=False) == set()


# --- what production requires ----------------------------------------------


def test_the_development_database_default_is_refused() -> None:
    """The one dangerous default: a well-known password on localhost."""
    assert "VOICEDESK_DATABASE_URL" in _settings(database_url=DEFAULT_DATABASE_URL)


def test_an_empty_database_url_is_refused() -> None:
    assert "VOICEDESK_DATABASE_URL" in _settings(database_url="  ")


def test_an_explicit_database_url_is_accepted() -> None:
    assert "VOICEDESK_DATABASE_URL" not in _settings()


def test_a_missing_model_key_is_refused() -> None:
    """There is no offline language model to fall back to."""
    assert "VOICEDESK_ANTHROPIC_API_KEY" in _settings(anthropic_api_key="")


def test_missing_speech_keys_are_not_refused() -> None:
    """Speech has an offline substitute that is real code. The model does not."""
    found = _settings(deepgram_api_key="", elevenlabs_api_key="")

    assert "VOICEDESK_DEEPGRAM_API_KEY" not in found
    assert "VOICEDESK_ELEVENLABS_API_KEY" not in found


def test_debug_is_refused_in_production() -> None:
    """It echoes every statement, including names, numbers and transcripts."""
    assert "VOICEDESK_DEBUG" in _settings(debug=True)


# --- what production requires once telephony is on -------------------------


def test_telephony_without_an_auth_token_is_refused() -> None:
    found = _settings(
        telephony_enabled=True,
        twilio_auth_token="",
        public_base_url="https://voicedesk.example",
    )

    assert "VOICEDESK_TWILIO_AUTH_TOKEN" in found


def test_telephony_without_a_public_base_url_is_refused() -> None:
    found = _settings(
        telephony_enabled=True, twilio_auth_token="token", public_base_url=""
    )

    assert "VOICEDESK_PUBLIC_BASE_URL" in found


def test_signature_validation_cannot_be_turned_off_in_production() -> None:
    found = _settings(
        telephony_enabled=True,
        twilio_auth_token="token",
        public_base_url="https://voicedesk.example",
        validate_twilio_signature=False,
    )

    assert "VOICEDESK_VALIDATE_TWILIO_SIGNATURE" in found


def test_the_unused_account_sid_is_not_required() -> None:
    """Nothing reads it. Requiring it would teach operators to ignore the list."""
    found = _settings(
        telephony_enabled=True,
        twilio_auth_token="token",
        public_base_url="https://voicedesk.example",
        twilio_account_sid="",
    )

    assert "VOICEDESK_TWILIO_ACCOUNT_SID" not in found


def test_a_complete_telephony_configuration_has_no_problems() -> None:
    assert (
        _settings(
            telephony_enabled=True,
            twilio_auth_token="token",
            public_base_url="https://voicedesk.example",
        )
        == set()
    )


# --- how it fails ----------------------------------------------------------


def test_check_raises_on_a_bad_production_configuration() -> None:
    with pytest.raises(ConfigurationError):
        check(_production(anthropic_api_key=""))


def test_check_is_silent_on_a_good_one() -> None:
    check(_production())


def test_every_problem_is_reported_at_once() -> None:
    """One variable per restart is a bad way to fix a deployment."""
    settings = _production(
        database_url=DEFAULT_DATABASE_URL,
        anthropic_api_key="",
        debug=True,
        telephony_enabled=True,
    )

    with pytest.raises(ConfigurationError) as raised:
        check(settings)

    message = str(raised.value)
    for setting in (
        "VOICEDESK_DATABASE_URL",
        "VOICEDESK_ANTHROPIC_API_KEY",
        "VOICEDESK_DEBUG",
        "VOICEDESK_TWILIO_AUTH_TOKEN",
        "VOICEDESK_PUBLIC_BASE_URL",
    ):
        assert setting in message


def test_the_failure_says_how_to_develop_instead() -> None:
    with pytest.raises(ConfigurationError, match="other than 'production'"):
        check(_production(anthropic_api_key=""))


def test_no_secret_value_appears_in_the_failure() -> None:
    """The message names variables. It never quotes what is in them."""
    settings = _production(
        telephony_enabled=True,
        twilio_auth_token="",
        public_base_url="",
        anthropic_api_key="",
        database_url="postgresql+psycopg://u:hunter2@db.example:5432/vd",
    )

    with pytest.raises(ConfigurationError) as raised:
        check(settings)

    assert "hunter2" not in str(raised.value)


# --- the application refuses to build --------------------------------------


def test_the_application_will_not_start_misconfigured(monkeypatch) -> None:
    from app.main import create_app

    monkeypatch.setattr(
        "app.main.get_settings", lambda: _production(anthropic_api_key="")
    )

    with pytest.raises(ConfigurationError):
        create_app()


def test_the_application_starts_when_production_is_configured(monkeypatch) -> None:
    from app.main import create_app

    monkeypatch.setattr("app.main.get_settings", lambda: _production())

    assert create_app() is not None
