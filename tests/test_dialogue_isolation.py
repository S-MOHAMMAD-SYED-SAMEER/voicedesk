"""The boundaries milestone 4 is not allowed to cross.

Parsed from source rather than grepped for, so a docstring may name the thing
a module must not import without tripping the check.
"""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parent.parent / "app"
DIALOGUE = APP / "dialogue"
PROVIDERS = APP / "providers"
TOOLS = APP / "tools"
CALENDAR = APP / "calendar"

VENDOR_SDKS = {"anthropic", "openai", "google", "cohere", "mistralai"}
TELEPHONY_AND_AUDIO = {
    "twilio",
    "websockets",
    "websocket",
    "sounddevice",
    "pyaudio",
    "wave",
    "audioop",
    "speech_recognition",
    "elevenlabs",
    "deepgram",
}


def _modules(path: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path.rglob("*.py"))


def _imports(source: str) -> set[str]:
    """Every module name a file imports, full dotted paths included."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _roots(source: str) -> set[str]:
    return {name.split(".")[0] for name in _imports(source)}


# --- the vendor SDK --------------------------------------------------------


def test_only_one_module_imports_the_anthropic_sdk() -> None:
    """The whole suite runs without an API key because of this."""
    importers = [
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if "anthropic" in _roots(module.read_text())
    ]
    assert importers == ["providers/anthropic_llm.py"]


def test_no_test_imports_the_anthropic_sdk_except_the_provider_test() -> None:
    tests = pathlib.Path(__file__).resolve().parent
    importers = [
        module.name
        for module in sorted(tests.glob("test_*.py"))
        if "anthropic" in _roots(module.read_text())
    ]
    assert importers == ["test_providers_llm.py"]


def test_the_providers_package_does_not_pull_in_an_sdk_on_import() -> None:
    """Importing `app.providers` must not cost a vendor dependency."""
    assert "anthropic" not in _imports((PROVIDERS / "__init__.py").read_text())
    assert "app.providers.anthropic_llm" not in _imports(
        (PROVIDERS / "__init__.py").read_text()
    )


def test_the_vendor_neutral_interface_names_no_vendor() -> None:
    assert not _roots((PROVIDERS / "llm.py").read_text()) & VENDOR_SDKS


def test_no_dialogue_module_imports_any_model_sdk() -> None:
    for module in _modules(DIALOGUE):
        assert not _roots(module.read_text()) & VENDOR_SDKS, module.name


# --- the layers below ------------------------------------------------------


def test_the_dialogue_layer_never_reaches_the_calendar() -> None:
    """Availability, hours and conflicts belong to milestone 2, through M3."""
    for module in _modules(DIALOGUE):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.calendar") for name in imports), module.name


def test_the_dialogue_layer_runs_no_queries_of_its_own() -> None:
    """It writes a transcript through the ORM; it reads nothing of its own.

    The one thing it reads — the service menu — goes through the tool layer's
    own helper, so there is a single definition of what "active" means.
    """
    for module in _modules(DIALOGUE):
        imported = {
            name
            for node in ast.walk(ast.parse(module.read_text()))
            if isinstance(node, ast.ImportFrom) and node.module == "sqlalchemy"
            for name in (alias.name for alias in node.names)
        }
        assert not imported & {"select", "text", "func", "insert", "update"}, module.name


def test_the_dialogue_layer_touches_only_the_transcript_models() -> None:
    """`turns` and `tool_calls` are milestone 4's to write. Nothing else is."""
    imported = set()
    for module in _modules(DIALOGUE):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "app.models":
                imported.update(alias.name for alias in node.names)
    assert imported == {"Call", "ToolCall", "Turn", "TurnRole"}


def test_tools_are_reached_only_through_the_registry() -> None:
    """No dialogue module imports a tool module directly.

    `app.tools` is the registry; `app.tools.base` is the shared vocabulary the
    registry itself is built from. A `app.tools.book_appointment` import would
    be a second way to call a tool, and the one that skips the guard.
    """
    for module in _modules(DIALOGUE):
        reached = {
            name
            for name in _imports(module.read_text())
            if name.startswith("app.tools")
        }
        assert reached <= {"app.tools", "app.tools.base"}, f"{module.name}: {reached}"


def test_the_executor_dispatches_through_get_tool() -> None:
    source = (DIALOGUE / "executor.py").read_text()
    assert "get_tool(" in source
    assert "from app.tools import" in source


# --- no audio, no telephony ------------------------------------------------


def test_nothing_in_milestone_four_imports_audio_or_telephony() -> None:
    for module in _modules(DIALOGUE) + _modules(PROVIDERS):
        assert not _roots(module.read_text()) & TELEPHONY_AND_AUDIO, module.name


# --- milestones 2 and 3 are untouched --------------------------------------


def test_the_tool_layer_still_knows_nothing_about_the_dialogue_layer() -> None:
    """Dependencies point one way: dialogue → tools → calendar."""
    for module in _modules(TOOLS):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.dialogue") for name in imports), module.name
        assert not any(name.startswith("app.providers") for name in imports), module.name


def test_the_calendar_still_knows_nothing_about_either() -> None:
    for module in _modules(CALENDAR):
        imports = _imports(module.read_text())
        for forbidden in ("app.dialogue", "app.providers", "app.tools"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )
