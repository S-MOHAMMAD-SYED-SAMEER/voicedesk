"""`python -m realeval.smoke` — exactly one real request, to prove the model ID and the request shape.

    python -m realeval.smoke --request-cap 200 --max-cost-usd 3.00 --input-usd-per-mtok 1 --output-usd-per-mtok 5

It sends **one** request through the budgeted model (so it is counted on the same ledger as the full run) with the real
system prompt, the clock line and the seven tool definitions, and one caller sentence. It does not run the tools, does
not loop and does not touch the database. It refuses to start if the ledger already holds a request, if the scenarios
were not frozen, or under any of the conditions a full run refuses. Nothing it prints contains the key.

Exit codes: 0 the request was answered, 2 it could not start, 3 the request failed (and nothing was retried).
"""

import sys
from collections.abc import Sequence
from zoneinfo import ZoneInfo

from app.dialogue.definitions import tool_definitions
from app.dialogue.prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_VERSION, clock_context
from app.evals.scenario import EVAL_NOW
from app.providers.llm import LanguageModel, Message, ModelError, ModelResponse
from realeval.__main__ import CANNOT_RUN, STOPPED, CannotRun, _model, _parser

CALLER_SENTENCE = "Hi, I'd like to book a haircut for tomorrow at ten. My name is Ada Lovelace and my number is 07700 900123."
MENU = "- Haircut\n- Beard trim"


def system_prompt() -> str:
    """What the dialogue layer sends on the first turn of a call, for the fixed evaluation moment and a two-service menu."""
    return f"{SYSTEM_PROMPT}\n{MENU}\n\n{clock_context(EVAL_NOW, ZoneInfo('UTC'))}\n"


def send_one(model: LanguageModel) -> ModelResponse:
    """The single request. One call to `respond`, whatever the model answers."""
    return model.respond(
        system=system_prompt(),
        messages=[Message(role="user", content=[{"type": "text", "text": CALLER_SENTENCE}])],
        tools=tool_definitions(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(["--provider", "anthropic", *(argv or sys.argv[1:])])
    try:
        model, budget, settings = _model(arguments)
        if budget.requests_used:
            raise CannotRun(
                f"the ledger already holds {budget.requests_used} request(s): the smoke request is the first one, "
                "and there is only one"
            )
    except CannotRun as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CANNOT_RUN

    print(f"Sending ONE request ({settings.dialogue_model}, system prompt {SYSTEM_PROMPT_VERSION}).", flush=True)
    try:
        response = send_one(model)
    except ModelError as exc:
        print(
            f"\nFAILED: {type(exc).__name__}: {str(exc)[:300]}\n"
            f"Latency: {model.requests[-1].get('latency_ms')} ms (timeout {settings.dialogue_timeout_seconds:g} s).\n"
            f"Ledger: {budget.requests_used} of {budget.request_cap} requests, {budget.describe_spend()}. "
            "Nothing was retried.",
            file=sys.stderr,
        )
        return STOPPED

    cost = budget.prices.cost(response.input_tokens, response.output_tokens)
    print(f"model requested : {settings.dialogue_model}")
    print(f"model resolved  : {response.model_name}")
    print(f"stop reason     : {response.stop_reason}")
    print(f"tool uses       : {[use.name for use in response.tool_uses] or 'none'}")
    print(f"tool arguments  : {[use.arguments for use in response.tool_uses] or 'none'}")
    print(f"reply text      : {response.text!r}")
    print(f"tokens          : {response.input_tokens} in, {response.output_tokens} out")
    print(f"latency         : {response.latency_ms} ms")
    print(f"cost            : USD {cost}")
    print(f"ledger          : {budget.requests_used} of {budget.request_cap} requests, {budget.describe_spend()} of USD {budget.cost_cap_usd}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
