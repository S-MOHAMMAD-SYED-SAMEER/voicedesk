"""Running the receptionist against a real model, with a hard cap on what it may spend.

    python -m realeval --list
    python -m realeval                              # offline: a stub model; verifies the machinery, measures nothing
    python -m realeval --provider anthropic --request-cap 200 --max-cost-usd 3.00 \\
        --input-usd-per-mtok 1 --output-usd-per-mtok 5

This package sits beside `app/`, not inside it, for two reasons: `app/evals` is offline by construction (no provider, no
network, no credential, and tests pin that), and the real-model run is something you run from a checkout, not something
the Docker image carries. It reuses everything model-independent from `app.evals` — the runner, the world, the trace,
the checks and the scoring — and changes only what is behind the model.

What it measures is **what a real model does with the scripted caller lines of `realeval/scenarios.py`**: a caller who
cannot adapt, over a small synthetic world, one run, one model. It is not an accuracy figure, and says nothing about
speech, telephony or real callers.
"""

#: Bumped when a scenario, an expectation or a rule changes in a way that makes two runs incomparable.
REAL_EVAL_VERSION = "r1"

__all__ = ["REAL_EVAL_VERSION"]
