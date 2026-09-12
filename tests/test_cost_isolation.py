"""The boundaries milestone 8 is not allowed to cross.

    transport ─► cost ─► models
                  │
                  └── pricing.py, and nothing else that knows a price

The cost layer is a leaf. Transports call it; it calls nobody back. Parsed
from source, so a docstring may name a thing a module must not import.
"""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parent.parent / "app"
COST = APP / "cost"
AUDIO = APP / "audio"
REALTIME = APP / "realtime"
DIALOGUE = APP / "dialogue"

VENDOR_SDKS = {"anthropic", "openai", "google", "cohere", "mistralai", "twilio"}
TRANSPORTS = ("app.audio", "app.realtime", "app.telephony", "app.api")


def _modules(path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path.rglob("*.py"))


def _imports(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _roots(source: str) -> set[str]:
    return {name.split(".")[0] for name in _imports(source)}


# --- the cost layer is a leaf ---------------------------------------------


def test_no_cost_module_reaches_a_transport() -> None:
    """It is called. It does not call back."""
    for module in _modules(COST):
        imports = _imports(module.read_text())
        for forbidden in TRANSPORTS:
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_no_cost_module_reaches_the_dialogue_or_the_calendar() -> None:
    for module in _modules(COST):
        imports = _imports(module.read_text())
        for forbidden in ("app.dialogue", "app.calendar", "app.tools"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_no_cost_module_imports_a_provider() -> None:
    """Usage arrives as numbers, not as a provider to be interrogated."""
    for module in _modules(COST):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.providers") for name in imports), (
            module.name
        )


def test_no_cost_module_imports_a_vendor_sdk() -> None:
    for module in _modules(COST):
        assert not _roots(module.read_text()) & VENDOR_SDKS, module.name


def test_no_cost_module_reaches_the_network() -> None:
    """A price is configured, never fetched."""
    for module in _modules(COST):
        assert not _roots(module.read_text()) & {
            "httpx",
            "requests",
            "urllib",
            "socket",
            "websockets",
        }, module.name


def test_the_cost_layer_touches_only_the_rows_it_owns() -> None:
    imported = set()
    for module in _modules(COST):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "app.models":
                imported.update(alias.name for alias in node.names)

    assert imported == {"Call", "CallCost", "CostComponent", "Turn"}


# --- pricing lives in exactly one module ----------------------------------


def _constructs(module: pathlib.Path, name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        for node in ast.walk(ast.parse(module.read_text()))
    )


def _assigns_to(module: pathlib.Path, attribute: str) -> bool:
    """Does this module set `something.<attribute>`, or name it in a keyword?"""
    for node in ast.walk(ast.parse(module.read_text())):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == attribute:
                    return True
        if isinstance(node, ast.keyword) and node.arg == attribute:
            return True
    return False


def test_only_the_recorder_constructs_a_cost_row() -> None:
    """One writer, so idempotency and totalling are decided in one place."""
    writers = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if _constructs(module, "CallCost")
    }

    assert writers == {"cost/recorder.py"}


def test_only_the_recorder_writes_the_call_total() -> None:
    """Every other mention of the column in `app/` is prose explaining it."""
    writers = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if _assigns_to(module, "total_cost_usd")
    }

    assert writers == {"cost/recorder.py"}


def test_the_model_declares_the_total_and_nothing_else_defines_it() -> None:
    declaring = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "total_cost_usd"
    }

    assert declaring == {"models/call.py"}


def test_nothing_outside_the_cost_layer_knows_a_rate() -> None:
    importers = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if "app.cost.pricing" in _imports(module.read_text())
    }

    assert importers <= {"cost/__init__.py", "cost/recorder.py"}


def test_no_module_outside_the_cost_layer_mentions_a_price_per_unit() -> None:
    """The conversions from an operator's units live in one file."""
    knows = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if "usd_per_unit" in module.read_text()
    }

    assert knows == {"cost/pricing.py"}


# --- the layers below stay where they were --------------------------------


def test_the_audio_layer_still_touches_no_database_model() -> None:
    """It gained usage fields, not a database connection."""
    for module in _modules(AUDIO):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.models") for name in imports), module.name


def test_the_audio_layer_does_not_know_what_anything_costs() -> None:
    for module in _modules(AUDIO):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.cost") for name in imports), module.name


def test_the_realtime_layer_does_not_know_what_anything_costs() -> None:
    for module in _modules(REALTIME):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.cost") for name in imports), module.name


def test_the_dialogue_layer_does_not_know_what_a_token_costs() -> None:
    """It counts them. Somebody else prices them."""
    for module in _modules(DIALOGUE):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.cost") for name in imports), module.name


def test_the_dialogue_layer_still_writes_only_its_own_rows() -> None:
    imported = set()
    for module in _modules(DIALOGUE):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "app.models":
                imported.update(alias.name for alias in node.names)

    assert imported == {"Call", "ToolCall", "Turn", "TurnRole"}


def test_only_the_latency_writer_still_touches_a_model_in_realtime() -> None:
    touching = {
        module.name
        for module in _modules(REALTIME)
        if any(name.startswith("app.models") for name in _imports(module.read_text()))
    }

    assert touching == {"latency.py"}


def test_the_model_provider_is_still_named_in_exactly_one_place() -> None:
    """The factory chooses a vendor, so the factory names it."""
    naming = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if '"anthropic"' in module.read_text()
    }

    assert naming == {"providers/factory.py"}


def test_the_carrier_is_still_named_only_where_the_carrier_is_spoken_to() -> None:
    naming = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if '"twilio"' in module.read_text()
    }

    assert naming == {"telephony/stream.py"}


def test_no_billing_machinery_was_added() -> None:
    """Accounting, not billing.

    Checked against names rather than prose: the cost package's docstring
    lists what is deliberately absent, which is the opposite of a problem.
    """
    forbidden = ("stripe", "invoice", "subscription", "checkout", "payment", "quota")
    for module in _modules(APP):
        tree = ast.parse(module.read_text())
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for name in names:
            for word in forbidden:
                assert word not in name.lower(), f"{module.name} defines {name}"
        for word in forbidden:
            assert word not in _roots(module.read_text()), module.name


def test_no_payment_dependency_was_added() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    text = root.read_text().lower()

    for word in ("stripe", "braintree", "paypal", "adyen"):
        assert word not in text
