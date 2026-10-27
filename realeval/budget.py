"""A hard cap on what a real-model run may spend, kept on disk.

The cap is for the whole piece of work (the one-request smoke, the scenario run, the recording), not per command: every
run that points at the same ledger draws on the same 200 requests and USD 3.00. A request is counted **before** it is
sent (a request that hangs, fails or is never answered still counts), and its cost is added when the answer arrives.

Nothing here retries. The first failure that means "stop" — a bad or missing credential, no credit, a rate limit, an
overloaded service, a timeout, a rejected request — stops the run at once: the next request would fail the same way,
and a service that is overloaded or rate-limited should be left alone. A refusal by the model is not a stop: it is the
model's answer to one request.

Adapter retries must be off (`VOICEDESK_DIALOGUE_MAX_RETRIES=0`, the default): a retry inside the SDK is an extra HTTP
request this ledger cannot see.

VoiceDesk ships no vendor prices, so the prices are given on the command line and recorded in the ledger.
"""

import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from app.providers.llm import (
    LanguageModel,
    Message,
    ModelError,
    ModelRefused,
    ModelResponse,
    ToolDefinition,
)

DEFAULT_LEDGER = Path(__file__).resolve().parent.parent / "var" / "real_eval_ledger.json"

#: Used only before any real cost has been seen: about 3 characters to a token, and the output ceiling.
CHARACTERS_PER_TOKEN = 3
ASSUMED_OUTPUT_TOKENS = 1024
TOKENS_PER_MILLION = Decimal(1_000_000)


class BudgetRefused(ModelError):
    """The cap is spent (or would be exceeded). Raised instead of sending; nothing was sent."""


class RunStopped(Exception):
    """The run stopped before every scenario was done, and why. A partial run is not a result."""

    def __init__(self, reason: str, done: int, total: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.done = done
        self.total = total


@dataclass(frozen=True)
class Prices:
    """USD per million tokens, as the operator gave them."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal

    def cost(self, input_tokens: int | None, output_tokens: int | None) -> Decimal | None:
        if input_tokens is None or output_tokens is None:
            return None
        return (
            Decimal(input_tokens) * self.input_per_mtok
            + Decimal(output_tokens) * self.output_per_mtok
        ) / TOKENS_PER_MILLION


class RunBudget:
    def __init__(
        self, *, request_cap: int, cost_cap_usd: Decimal, prices: Prices, path: Path | None = None
    ) -> None:
        self.request_cap = request_cap
        self.cost_cap_usd = cost_cap_usd
        self.prices = prices
        self.path = path or DEFAULT_LEDGER

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"entries": []}

    def _write(self, ledger: dict) -> None:
        ledger["request_cap"] = self.request_cap
        ledger["cost_cap_usd"] = str(self.cost_cap_usd)
        ledger["prices_usd_per_mtok"] = {
            "input": str(self.prices.input_per_mtok),
            "output": str(self.prices.output_per_mtok),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    @property
    def requests_used(self) -> int:
        return len(self._read()["entries"])

    @staticmethod
    def _counted(entry: dict) -> Decimal:
        """What an entry counts against the cap: its cost from the tokens, or, when no answer came back, its maximum.

        A request that timed out, failed or was refused may still have been billed, so it is never counted as zero:
        it counts at the most it could have cost (the estimated input tokens plus `max_tokens` at the given prices).
        """
        if entry.get("cost_usd"):
            return Decimal(entry["cost_usd"])
        if entry.get("max_cost_usd"):
            return Decimal(entry["max_cost_usd"])
        return Decimal(0)

    @property
    def spent_usd(self) -> Decimal:
        """Cost counted against the cap: real costs, plus the maximum of every request whose cost is unknown."""
        return sum((self._counted(entry) for entry in self._read()["entries"]), Decimal(0))

    @property
    def unknown_requests(self) -> int:
        """Requests that were sent but whose cost is not known (no answer came back)."""
        return sum(1 for entry in self._read()["entries"] if not entry.get("cost_usd"))

    def describe_spend(self) -> str:
        unknown = self.unknown_requests
        if not unknown:
            return f"USD {self.spent_usd:.6f}"
        return f"USD {self.spent_usd:.6f} counted, of which {unknown} request(s) of unknown cost at their maximum"

    def latencies_ms(self) -> list[int]:
        return [e["latency_ms"] for e in self._read()["entries"] if e.get("latency_ms") is not None]

    def largest_cost_seen(self) -> Decimal:
        costs = [Decimal(e["cost_usd"]) for e in self._read()["entries"] if e.get("cost_usd")]
        return max(costs, default=Decimal(0))

    def reserve(self, label: str, projected_cost: Decimal) -> int:
        """Count one request before it is sent. Returns its index; refuses if the cap would be passed."""
        ledger = self._read()
        used = len(ledger["entries"])
        if used >= self.request_cap:
            raise BudgetRefused(f"the request cap of {self.request_cap} is spent ({used} used)")
        spent = sum((self._counted(e) for e in ledger["entries"]), Decimal(0))
        if spent + projected_cost > self.cost_cap_usd:
            raise BudgetRefused(
                f"another request (about USD {projected_cost:.6f}) would pass the USD {self.cost_cap_usd} cap "
                f"({spent:.6f} spent)"
            )
        ledger["entries"].append(
            {
                "at": datetime.now(UTC).isoformat(),
                "label": label,
                "cost_usd": None,
                # The most this request could cost; it counts against the cap until the real cost is known.
                "max_cost_usd": str(projected_cost),
                "tokens": None,
            }
        )
        self._write(ledger)
        return used

    def record(
        self,
        index: int,
        cost: Decimal | None,
        tokens: tuple[int | None, int | None],
        model: str,
        *,
        latency_ms: int | None = None,
        outcome: str = "answered",
    ) -> None:
        ledger = self._read()
        entry = ledger["entries"][index]
        entry["cost_usd"] = str(cost) if cost is not None else None
        entry["tokens"] = {"input": tokens[0], "output": tokens[1]}
        entry["model"] = model
        entry["latency_ms"] = latency_ms
        entry["outcome"] = outcome
        if cost is None:
            entry["cost_note"] = "unknown, at most max_cost_usd (a request that got no answer may still be billed)"
        self._write(ledger)


def percentile(values: Sequence[int], fraction: float) -> int | None:
    """Nearest-rank percentile; `None` for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def latency_summary(values: Sequence[int]) -> dict[str, int | None]:
    """Request latency, reported and never graded: a receptionist is not scored on it."""
    return {"requests": len(values), "p50_ms": percentile(values, 0.50), "p95_ms": percentile(values, 0.95)}


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


class BudgetedModel:
    """Counts every `respond()` against the budget before the inner model sees it, and stops at the first stop-rule error."""

    def __init__(self, inner: LanguageModel, budget: RunBudget, *, label: str = "real-eval") -> None:
        self._inner = inner
        self.budget = budget
        self._label = label
        self.requests: list[dict] = []
        #: Set by the first failure that means "stop"; the runner checks it after every turn.
        self.halt_reason: str | None = None

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def model_name(self) -> str:
        return getattr(self._inner, "model_name", "")

    def _projected_cost(
        self, system: str, messages: Sequence[Message], tools: Sequence[ToolDefinition] = ()
    ) -> Decimal:
        seen = self.budget.largest_cost_seen()
        characters = (
            len(system)
            + sum(len(json.dumps(message.content, default=str)) for message in messages)
            + sum(len(json.dumps(tool, default=str)) for tool in tools)
        )
        tokens_in = characters // CHARACTERS_PER_TOKEN
        estimate = self.budget.prices.cost(tokens_in, ASSUMED_OUTPUT_TOKENS) or Decimal(0)
        return max(seen, estimate)

    def respond(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolDefinition]
    ) -> ModelResponse:
        if self.halt_reason is not None:
            raise BudgetRefused(f"the run was stopped ({self.halt_reason}); nothing was sent")
        try:
            index = self.budget.reserve(self._label, self._projected_cost(system, messages, tools))
        except BudgetRefused as exc:
            self.halt_reason = str(exc)
            raise
        request: dict = {"index": index}
        self.requests.append(request)
        started = time.monotonic()
        try:
            response = self._inner.respond(system=system, messages=messages, tools=tools)
        except ModelRefused:
            # A refusal is the model's answer to this request; it says nothing about the next one.
            request["latency_ms"] = _elapsed_ms(started)
            self.budget.record(
                index, None, (None, None), self.model_name, latency_ms=request["latency_ms"], outcome="refused"
            )
            raise
        except ModelError as exc:
            self.halt_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            request["latency_ms"] = _elapsed_ms(started)
            self.budget.record(
                index, None, (None, None), self.model_name, latency_ms=request["latency_ms"], outcome=type(exc).__name__
            )
            raise
        request["latency_ms"] = response.latency_ms if response.latency_ms is not None else _elapsed_ms(started)
        self.budget.record(
            index,
            self.budget.prices.cost(response.input_tokens, response.output_tokens),
            (response.input_tokens, response.output_tokens),
            response.model_name,
            latency_ms=request["latency_ms"],
        )
        return response


__all__ = [
    "DEFAULT_LEDGER",
    "BudgetRefused",
    "BudgetedModel",
    "Prices",
    "RunBudget",
    "RunStopped",
    "latency_summary",
    "percentile",
]
