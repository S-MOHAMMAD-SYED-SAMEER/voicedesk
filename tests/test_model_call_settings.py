"""The model call is bounded: 15 seconds, no hidden retries, and a key from either name; the requirements install on Windows."""

import pathlib

import pytest

from app.config import Settings
from app.providers.anthropic_llm import AnthropicLanguageModel

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_defaults_are_fifteen_seconds_and_no_retries() -> None:
    settings = Settings(_env_file=None)

    assert settings.dialogue_timeout_seconds == 15.0
    assert settings.dialogue_max_retries == 0
    assert settings.holding_line == "One moment, let me check."
    assert settings.holding_line_after_seconds == 2.0
    assert settings.dialogue_model == "claude-haiku-4-5"


def test_the_client_is_built_with_the_timeout_and_zero_retries() -> None:
    """Said explicitly: the SDK would otherwise retry twice, tripling the silence on a call."""
    model = AnthropicLanguageModel(settings=Settings(_env_file=None, anthropic_api_key="sk-test"))

    assert model._client.max_retries == 0
    assert model._client.timeout == 15.0


def test_the_timeout_and_retries_are_configurable() -> None:
    settings = Settings(_env_file=None, anthropic_api_key="sk-test", dialogue_timeout_seconds=7, dialogue_max_retries=1)
    model = AnthropicLanguageModel(settings=settings)

    assert (model._client.timeout, model._client.max_retries) == (7.0, 1)


def test_the_key_is_read_from_the_voicedesk_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("VOICEDESK_ANTHROPIC_API_KEY", "sk-voicedesk")

    assert Settings(_env_file=None).anthropic_api_key == "sk-voicedesk"


def test_the_sdks_own_name_works_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICEDESK_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-sdk")

    assert Settings(_env_file=None).anthropic_api_key == "sk-sdk"


def test_the_voicedesk_name_wins_when_both_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICEDESK_ANTHROPIC_API_KEY", "sk-voicedesk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-sdk")

    assert Settings(_env_file=None).anthropic_api_key == "sk-voicedesk"


def test_the_key_is_never_in_a_settings_repr_that_a_log_would_print() -> None:
    """Names only: a key must not appear when settings are printed by accident."""
    settings = Settings(_env_file=None, anthropic_api_key="sk-secret-value")

    assert "sk-secret-value" not in repr(settings.model_dump(exclude={"anthropic_api_key"}))


# --- the pinned requirements -------------------------------------------------


def _line(prefix: str) -> str:
    return next(
        line
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.startswith(prefix)
    )


def test_uvloop_is_pinned_only_where_it_exists() -> None:
    """uvloop has no Windows build, so an unconditional pin made `pip install -r requirements.txt` fail there."""
    assert _line("uvloop==").endswith('; sys_platform != "win32"')


def test_every_other_pin_is_unconditional() -> None:
    conditional = [
        line
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if ";" in line and not line.startswith("#")
    ]

    assert [line.split("==")[0] for line in conditional] == ["uvloop"]


def test_ruff_is_a_development_dependency_and_migrations_are_excluded() -> None:
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert any(item.startswith("ruff") for item in config["project"]["optional-dependencies"]["dev"])
    assert "alembic" in config["tool"]["ruff"]["extend-exclude"]
