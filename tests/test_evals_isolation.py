"""The boundaries milestone 9 is not allowed to cross.

    app.evals ──► everything below it, read-only where it matters
    app.evals ◄── nothing

The evaluator stands where a transport stands: it calls the system, and the
system knows nothing about it. If production code ever imported `app.evals`,
the thing being measured would include the measurer.

Also here: proof that milestone 9 changed nothing it promised not to. Frozen
modules are checked against the commit M9 started from, so "unchanged" is a
fact about the repository rather than an intention.
"""

import ast
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = ROOT.parent
APP = ROOT / "app"
EVALS = APP / "evals"

# Where M9 began. Frozen files are diffed against it.
M8_COMMIT = "f18bd15"

VENDOR_SDKS = {"anthropic", "openai", "google", "cohere", "mistralai", "twilio"}
NETWORK = {"httpx", "requests", "urllib", "socket", "websockets", "http"}


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


def _changed_since(commit: str, path: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", commit, "--", path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.split() if line]


# --- the evaluator is a leaf -----------------------------------------------


def test_no_production_module_imports_the_evaluator() -> None:
    """The thing being measured must not know it is being measured."""
    importers = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if not module.is_relative_to(EVALS)
        and any(name.startswith("app.evals") for name in _imports(module.read_text()))
    }

    assert importers == set()


def test_the_evaluator_imports_nothing_from_the_tests() -> None:
    """`app/` may never depend on `tests/`, evaluator included."""
    for module in _modules(EVALS):
        imports = _imports(module.read_text())
        assert not any(
            name == "tests" or name.startswith("tests.") for name in imports
        ), module.name
        assert "conftest" not in module.read_text(), module.name


def test_the_evaluator_imports_no_vendor_sdk() -> None:
    for module in _modules(EVALS):
        assert not _roots(module.read_text()) & VENDOR_SDKS, module.name


def test_the_evaluator_reaches_no_network() -> None:
    """Offline by construction, not by configuration."""
    for module in _modules(EVALS):
        assert not _roots(module.read_text()) & NETWORK, module.name


def test_the_evaluator_imports_no_provider_implementation_but_the_offline_ones() -> None:
    allowed = {
        "app.providers.offline_streaming",
        "app.providers.speech",
        "app.providers.streaming_tts",
        "app.providers.llm",
    }
    for module in _modules(EVALS):
        for name in _imports(module.read_text()):
            if name.startswith("app.providers"):
                assert name in allowed, f"{module.name} imports {name}"


def test_the_evaluator_names_no_credential() -> None:
    for module in _modules(EVALS):
        lowered = module.read_text().lower()
        for word in ("api_key", "auth_token", "secret"):
            assert word not in lowered, f"{module.name} mentions {word}"


def test_the_evaluator_uses_no_llm_as_a_judge() -> None:
    """Every verdict is a comparison of values, and must stay one."""
    judge = APP / "evals" / "checks.py"
    imports = _imports(judge.read_text())

    assert not any(name.startswith("app.providers.anthropic") for name in imports)
    assert "app.evals.model" not in imports


# --- the evaluator reaches the real system ---------------------------------


def test_the_evaluator_drives_the_real_dialogue_layer() -> None:
    """A second dialogue implementation would measure itself."""
    source = (EVALS / "runner.py").read_text()

    assert "app.dialogue" in _imports(source)
    assert "Conversation(" in source


def test_the_evaluator_builds_its_world_through_the_real_calendar() -> None:
    source = (EVALS / "world.py").read_text()

    assert "app.calendar" in _imports(source)
    assert "calendar.book(" in source


def test_the_evaluator_defines_no_tool_of_its_own() -> None:
    """It observes the six that exist; it does not add a seventh."""
    from app.tools import TOOLS

    for module in _modules(EVALS):
        tree = ast.parse(module.read_text())
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not defined & set(TOOLS), module.name


def test_the_evaluator_computes_no_availability_of_its_own() -> None:
    for module in _modules(EVALS):
        source = module.read_text().lower()
        assert "available_slots" not in source, module.name
        assert "is_available" not in source, module.name


# --- milestone 8 is untouched ----------------------------------------------


def test_the_cost_layer_is_unchanged() -> None:
    assert _changed_since(M8_COMMIT, "voicedesk/app/cost") == []


def test_the_evaluator_adds_no_pricing() -> None:
    for module in _modules(EVALS):
        source = module.read_text().lower()
        assert "usd_per" not in source, module.name
        assert "pricebook" not in source, module.name


def test_the_evaluator_records_cost_only_through_the_milestone_8_recorder() -> None:
    source = (EVALS / "runner.py").read_text()

    assert "record_turn_cost" in source
    assert "app.cost.pricing" not in _imports(source)


def test_the_evaluator_writes_no_cost_row_of_its_own() -> None:
    for module in _modules(EVALS):
        tree = ast.parse(module.read_text())
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CallCost"
            for node in ast.walk(tree)
        ), module.name


# --- milestone 7 and the layers below are untouched ------------------------


@pytest.mark.parametrize(
    "path",
    [
        "voicedesk/app/realtime",
        "voicedesk/app/audio",
        "voicedesk/app/calendar",
        "voicedesk/app/tools",
        "voicedesk/app/dialogue",
        "voicedesk/app/telephony",
        "voicedesk/app/models",
        "voicedesk/app/providers",
        "voicedesk/alembic",
    ],
)
def test_a_frozen_area_is_unchanged(path: str) -> None:
    assert _changed_since(M8_COMMIT, path) == []


def test_docintel_is_untouched() -> None:
    assert _changed_since(M8_COMMIT, "docintel") == []


def test_only_the_approved_production_files_changed() -> None:
    """Milestone 9 touches one production file: the one new setting."""
    changed = set(_changed_since(M8_COMMIT, "voicedesk/app"))
    outside = {name for name in changed if not name.startswith("voicedesk/app/evals/")}

    assert outside == {"voicedesk/app/config.py"}


# --- no migration ----------------------------------------------------------


def test_there_are_still_exactly_three_migrations() -> None:
    versions = ROOT / "alembic" / "versions"

    assert len(list(versions.glob("*.py"))) == 3


def test_the_evaluator_adds_no_table() -> None:
    """The trace is in memory. The system under test writes its own rows."""
    from app.db.base import Base

    assert set(Base.metadata.tables) == {
        "calls",
        "turns",
        "tool_calls",
        "appointments",
        "services",
        "business_hours",
        "call_costs",
    }


def test_the_evaluator_declares_no_model() -> None:
    for module in _modules(EVALS):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {
                    base.id for base in node.bases if isinstance(base, ast.Name)
                }
                assert "Base" not in bases, f"{module.name}.{node.name}"


# --- no new dependency -----------------------------------------------------


def test_no_dependency_was_added() -> None:
    assert _changed_since(M8_COMMIT, "voicedesk/pyproject.toml") == []


def test_the_evaluator_uses_only_what_is_already_installed() -> None:
    allowed = {
        "app",
        "alembic",
        "sqlalchemy",
        "anyio",
        "argparse",
        "json",
        "statistics",
        "dataclasses",
        "decimal",
        "datetime",
        "pathlib",
        "logging",
        "re",
        "sys",
        "os",
        "uuid",
        "math",
        "struct",
        "typing",
        "collections",
        "contextlib",
        "zoneinfo",
    }
    for module in _modules(EVALS):
        assert _roots(module.read_text()) <= allowed, module.name


# --- no HTTP surface -------------------------------------------------------


def test_the_evaluator_serves_nothing() -> None:
    for module in _modules(EVALS):
        roots = _roots(module.read_text())
        assert "fastapi" not in roots, module.name
        assert "starlette" not in roots, module.name


def test_the_application_gained_no_route() -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    paths = set(TestClient(create_app()).app.openapi()["paths"])

    assert not any("eval" in path for path in paths)
    assert paths == {"/health", "/harness", "/telephony/voice"}
