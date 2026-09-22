"""The boundary milestone 10 was given, checked against the repository.

This is a hardening milestone. It was allowed to touch a named set of files
for named reasons, and nothing else — so "nothing else changed" is asserted
here as a fact about git rather than an intention in a commit message.

Everything is diffed against `dd19e04`, the commit milestone 10 started from.
"""

import ast
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
# This repository's own root. Before VoiceDesk was extracted from the Prjs
# monorepo, ROOT was the voicedesk/ subdirectory and REPO was one level up,
# at the monorepo root. Extraction promoted voicedesk/ to be the repository
# root itself, so the git root and the package root now coincide.
REPO = ROOT
APP = ROOT / "app"

# Where milestone 10 began. This is the hash git-filter-repo gave the
# milestone-9 "add evaluation system" commit when VoiceDesk was extracted to
# its own repository — the commit's content, author, date and message are
# unchanged; only its tree (now rooted here instead of at voicedesk/ inside
# the monorepo) was rewritten. It was dd19e04 in the Prjs monorepo.
M9_COMMIT = "5f0a330"

# The production modules milestone 10 was approved to change, each for one
# reason. Anything outside this set showing up in the diff is scope creep.
APPROVED = {
    # The refused-reschedule rollback: the defect milestone 9 measured.
    "app/calendar/service.py",
    # Production settings, timeouts, and the two properties the guards read.
    "app/config.py",
    # Preflight, draining, the closed documentation surface, the error
    # boundary.
    "app/main.py",
    # Refused in production, and admitted through the same counter as a call.
    "app/api/harness.py",
    # A connect timeout on the engine.
    "app/db/session.py",
    # A request timeout on the model client.
    "app/providers/anthropic_llm.py",
    # Connect and idle timeouts on the two streaming sockets.
    "app/providers/deepgram_stream_stt.py",
    "app/providers/elevenlabs_stream_tts.py",
    # The stream token, admission, frame and duration limits.
    "app/telephony/stream.py",
    "app/telephony/twiml.py",
    "app/telephony/webhook.py",
}

# What milestone 10 added. New files, so nothing they do can regress
# anything that was already working.
ADDED = {
    "app/api/ready.py",
    "app/logging.py",
    "app/preflight.py",
    "app/providers/streaming.py",
    "app/retention.py",
    "app/runtime.py",
    "app/telephony/stream_token.py",
}


def _changed_since(commit: str, path: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", commit, "--", path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(line for line in result.stdout.split() if line)


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


# --- only the approved files changed ---------------------------------------


def test_no_production_file_outside_the_approved_set_changed() -> None:
    changed = set(_changed_since(M9_COMMIT, "app"))

    assert changed <= APPROVED | ADDED, changed - (APPROVED | ADDED)


def test_every_approved_change_was_actually_made() -> None:
    """The other direction: a file listed as changed that is not would mean
    this list had drifted away from the milestone it describes."""
    changed = set(_changed_since(M9_COMMIT, "app"))

    assert APPROVED <= changed, APPROVED - changed


def test_each_new_module_is_there() -> None:
    for name in ADDED:
        assert (REPO / name).is_file(), name


# --- the frozen areas ------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        # The tool semantics, the prompt and the tool loop.
        "app/tools",
        "app/dialogue",
        # The VAD algorithm and barge-in.
        "app/realtime",
        "app/audio",
        # Pricing.
        "app/cost",
        # The evaluator's semantics.
        "app/evals",
        # The schema and its migrations.
        "app/models",
        "alembic",
    ],
)
def test_a_frozen_area_is_unchanged(path: str) -> None:
    assert _changed_since(M9_COMMIT, path) == []


# There is no standalone equivalent of "docintel is untouched": docintel was
# a sibling project in the Prjs monorepo, and this repository does not
# contain it at all post-extraction. The invariant that boundary protected
# no longer has anything to check, so the assertion was removed rather than
# kept as a vacuous pass.


# --- no schema change ------------------------------------------------------


def test_there_are_still_exactly_three_migrations() -> None:
    versions = ROOT / "alembic" / "versions"

    assert len(list(versions.glob("*.py"))) == 3


def test_no_table_was_added() -> None:
    """Admission, draining and the stream token are all in memory. Retention
    deletes rows; it does not record that it did."""
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


def test_no_new_module_declares_a_model() -> None:
    for name in sorted(ADDED):
        tree = ast.parse((REPO / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
                assert "Base" not in bases, f"{name}.{node.name}"


# --- no new dependency -----------------------------------------------------


def test_no_dependency_was_added_or_moved() -> None:
    """The pinned set in `requirements.txt` is a record of what was already
    installed, not a reason to install anything."""
    assert _changed_since(M9_COMMIT, "pyproject.toml") == []


def test_the_new_modules_use_only_what_is_already_installed() -> None:
    allowed = {
        "app",
        "alembic",
        "sqlalchemy",
        "fastapi",
        "pydantic",
        "anyio",
        "argparse",
        "base64",
        "collections",
        "contextlib",
        "dataclasses",
        "datetime",
        "functools",
        "hashlib",
        "hmac",
        "json",
        "logging",
        "pathlib",
        "threading",
        "time",
        "typing",
        "urllib",
    }
    for name in sorted(ADDED):
        source = (REPO / name).read_text()
        roots = {imported.split(".")[0] for imported in _imports(source)}
        assert roots <= allowed, f"{name}: {roots - allowed}"


# --- no new capability -----------------------------------------------------


def test_the_route_set_gained_exactly_one_path() -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    paths = set(TestClient(create_app()).app.openapi()["paths"])

    assert paths == {"/health", "/ready", "/harness", "/telephony/voice"}


def test_the_six_tools_are_still_the_six_tools() -> None:
    """No tool was added, removed or renamed."""
    from app.tools import TOOLS

    assert set(TOOLS) == {
        "check_availability",
        "book_appointment",
        "reschedule",
        "cancel",
        "take_message",
        "transfer_to_human",
    }


def test_no_new_provider_arrived() -> None:
    from app.config import STREAMING_STT_PROVIDERS, STT_PROVIDERS, TTS_PROVIDERS

    assert set(STT_PROVIDERS) == {"offline", "deepgram"}
    assert set(TTS_PROVIDERS) == {"offline", "elevenlabs"}
    assert set(STREAMING_STT_PROVIDERS) == {"offline", "deepgram"}


# Capability names, not words. `CallDirection.OUTBOUND` has been in the
# schema since milestone 1 and is a column value, not an ability to place a
# call, so what is checked is the verb rather than the noun.
NOT_IN_SCOPE = (
    "sms",
    "place_call",
    "dial_out",
    "billing",
    "invoice",
    "tenant",
    "kubernetes",
    "terraform",
)


def _identifiers(tree: ast.AST) -> set[str]:
    """Every name the code actually uses, prose excluded.

    Docstrings are where this repository explains what it deliberately did
    not build — "cost tracking, not billing" — so a scan of the raw text
    would fail on the sentence saying the thing is absent.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
    return found


@pytest.mark.parametrize("absent", NOT_IN_SCOPE)
def test_nothing_outside_the_milestone_appeared(absent: str) -> None:
    """The explicit not-in-scope list, asserted rather than remembered."""
    for module in _modules(APP):
        names = _identifiers(ast.parse(module.read_text()))
        offending = {name for name in names if absent in name.lower()}
        assert not offending, f"{module.name}: {offending}"


def test_no_configuration_variable_was_renamed() -> None:
    """An operator's existing environment must keep working."""
    import subprocess as _subprocess

    from app.config import Settings

    before = _subprocess.run(
        ["git", "show", f"{M9_COMMIT}:app/config.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tree = ast.parse(before)
    settings = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
    )
    was = {
        node.target.id
        for node in settings.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert was <= set(Settings.model_fields), was - set(Settings.model_fields)


# --- the new modules are leaves --------------------------------------------


def test_nothing_in_the_application_imports_the_retention_command() -> None:
    """A command that deletes calls must not be reachable from a request."""
    for module in _modules(APP):
        if module.name == "retention.py":
            continue
        assert "app.retention" not in _imports(module.read_text()), module.name


def test_the_readiness_probe_is_not_imported_by_anything_but_the_app() -> None:
    importers = {
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if "app.api.ready" in _imports(module.read_text())
    }

    assert importers <= {"main.py", "api/__init__.py"}


def test_the_preflight_imports_no_provider() -> None:
    """It reads configuration. It must not reach an account to check one."""
    source = (APP / "preflight.py").read_text()

    assert not any(
        name.startswith("app.providers") for name in _imports(source)
    )
