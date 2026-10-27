"""The smoke request: exactly one, counted, and nothing else. No network, no database."""

import argparse
from decimal import Decimal

import pytest

from app.config import Settings, get_settings
from app.providers.llm import ModelUnavailable
from realeval import __main__ as cli
from realeval import smoke
from realeval.budget import BudgetedModel, Prices, RunBudget

from .conftest import FakeModel, say, use_tools

PRICES = Prices(Decimal("1"), Decimal("5"))
ARGS = ["--request-cap", "200", "--max-cost-usd", "3.00", "--input-usd-per-mtok", "1", "--output-usd-per-mtok", "5"]


def _budget(tmp_path) -> RunBudget:
    return RunBudget(request_cap=200, cost_cap_usd=Decimal("3.00"), prices=PRICES, path=tmp_path / "ledger.json")


def _patched(monkeypatch: pytest.MonkeyPatch, tmp_path, fake: FakeModel, *, used: int = 0):
    budget = _budget(tmp_path)
    for _ in range(used):
        budget.reserve("earlier", Decimal("0.001"))
    model = BudgetedModel(fake, budget)
    settings = Settings(_env_file=None)
    monkeypatch.setattr(smoke, "_model", lambda arguments: (model, budget, settings))
    return budget


def test_the_request_is_the_real_prompt_the_seven_tools_and_one_caller_sentence() -> None:
    fake = FakeModel(say("Certainly."))

    smoke.send_one(fake)

    (request,) = fake.requests
    assert "Rules you must not break" in request["system"]
    assert "Haircut" in request["system"] and "2026-03-01" in request["system"]
    assert len(request["tools"]) == 7
    assert [m.content[0]["text"] for m in request["messages"]] == [smoke.CALLER_SENTENCE]


def test_one_request_is_sent_counted_and_reported(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    fake = FakeModel(use_tools(("check_availability", {"service_name": "Haircut", "day": "2026-03-02"})))
    budget = _patched(monkeypatch, tmp_path, fake)

    code = smoke.main(ARGS)

    assert code == 0
    assert fake.call_count == 1 and budget.requests_used == 1
    out = capsys.readouterr().out
    assert "check_availability" in out and "1 of 200 requests" in out
    assert "sk-" not in out


def test_the_tools_the_model_asks_for_are_not_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A tool use ends the smoke: there is no loop, so there is no second request."""
    fake = FakeModel(
        use_tools(("check_availability", {"service_name": "Haircut", "day": "2026-03-02"})),
        say("This would be a second request."),
    )
    _patched(monkeypatch, tmp_path, fake)

    smoke.main(ARGS)

    assert fake.call_count == 1


def test_a_second_smoke_is_refused_before_anything_is_sent(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    fake = FakeModel(say("never sent"))
    _patched(monkeypatch, tmp_path, fake, used=1)

    code = smoke.main(ARGS)

    assert code == cli.CANNOT_RUN
    assert fake.call_count == 0
    assert "only one" in capsys.readouterr().err


def test_a_failed_request_is_reported_and_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    fake = FakeModel(raises=ModelUnavailable("overloaded"))
    budget = _patched(monkeypatch, tmp_path, fake)

    code = smoke.main(ARGS)

    assert code == cli.STOPPED
    assert fake.call_count == 1 and budget.requests_used == 1
    assert "Nothing was retried" in capsys.readouterr().err


def test_the_smoke_refuses_to_start_before_the_scenarios_are_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: get_settings().model_copy(update={"anthropic_api_key": "sk-test"}))
    pending = tmp_path / "EVAL.md"
    pending.write_text("real-eval-sha256: `pending approval`")
    monkeypatch.setattr(cli, "EVAL_DOC", pending)

    code = smoke.main([*ARGS, "--ledger", str(tmp_path / "ledger.json")])

    assert code == cli.CANNOT_RUN
    assert "not been frozen" in capsys.readouterr().err
    assert not (tmp_path / "ledger.json").exists()


def test_the_smoke_is_an_anthropic_run_whatever_is_typed() -> None:
    namespace = smoke._parser().parse_args(["--provider", "anthropic", *ARGS])

    assert isinstance(namespace, argparse.Namespace) and namespace.provider == "anthropic"
