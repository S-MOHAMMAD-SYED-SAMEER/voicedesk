"""The boundaries milestone 6 is not allowed to cross.

    telephony → audio → dialogue → tools → calendar → database

and never telephony → calendar, telephony → a tool, or telephony → a model.
Parsed from source, so a docstring may name a thing a module must not import.
"""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parent.parent / "app"
TELEPHONY = APP / "telephony"
AUDIO = APP / "audio"
DIALOGUE = APP / "dialogue"
TOOLS = APP / "tools"
CALENDAR = APP / "calendar"
TELEPHONY_AUDIO = AUDIO / "telephony.py"
EVENTS = TELEPHONY / "events.py"

VENDOR_SDKS = {"anthropic", "openai", "google", "cohere", "mistralai", "twilio"}


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


# --- telephony is an adapter, not a second product -------------------------


def test_no_telephony_module_reaches_the_calendar() -> None:
    for module in _modules(TELEPHONY):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.calendar") for name in imports), module.name


def test_no_telephony_module_reaches_the_tools() -> None:
    """It has no business booking anything. That is what the dialogue is for."""
    for module in _modules(TELEPHONY):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.tools") for name in imports), module.name


def test_no_telephony_module_imports_a_vendor_sdk() -> None:
    """Including the carrier's own: this is JSON over a socket, nothing more."""
    for module in _modules(TELEPHONY):
        assert not _roots(module.read_text()) & VENDOR_SDKS, module.name


def test_no_telephony_module_imports_a_model_implementation() -> None:
    for module in _modules(TELEPHONY):
        imports = _imports(module.read_text())
        for forbidden in (
            "app.providers.anthropic_llm",
            "app.providers.deepgram_stt",
            "app.providers.elevenlabs_tts",
        ):
            assert forbidden not in imports, f"{module.name} imports {forbidden}"


def test_telephony_reaches_the_dialogue_only_through_voice_session() -> None:
    """One dialogue path. The browser takes the same one."""
    imported = set()
    for module in _modules(TELEPHONY):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "app.dialogue":
                imported.update(alias.name for alias in node.names)
    assert imported == {"Conversation"}
    assert "VoiceSession" in (TELEPHONY / "stream.py").read_text()


def test_telephony_selects_providers_through_the_factory() -> None:
    assert "app.providers.factory" in _imports((TELEPHONY / "stream.py").read_text())


def test_telephony_touches_only_the_call_model() -> None:
    """Turns and tool calls are the dialogue layer's to write, not this one's."""
    imported = set()
    for module in _modules(TELEPHONY):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "app.models":
                imported.update(alias.name for alias in node.names)
    assert imported == {"Call", "CallDirection"}


# --- the carrier's wire format lives in one place --------------------------


def test_only_the_event_module_knows_the_carriers_json() -> None:
    """Everything else speaks in dataclasses."""
    markers = ("streamSid", "callSid", "mediaFormat", "audio/x-mulaw")
    for marker in markers:
        knows = [
            module.relative_to(APP).as_posix()
            for module in _modules(APP)
            if marker in module.read_text()
        ]
        assert knows == ["telephony/events.py"], f"{marker!r} appears in {knows}"


def test_the_audio_adapter_knows_nothing_about_the_carrier() -> None:
    """It converts µ-law. Who is carrying it is not its business."""
    source = TELEPHONY_AUDIO.read_text()

    assert "twilio" not in source.lower()
    assert not any(name.startswith("app.telephony") for name in _imports(source))
    assert "json" not in _roots(source)


def test_the_event_module_does_no_audio_conversion() -> None:
    """Base64 is transport. µ-law is audio, and lives one layer down."""
    source = EVENTS.read_text()

    assert "mulaw_decode" not in source
    assert not any(name.startswith("app.audio") for name in _imports(source))


# --- the layers below are unchanged ----------------------------------------


def test_the_audio_layer_still_knows_nothing_about_telephony_protocol() -> None:
    for module in _modules(AUDIO):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.telephony") for name in imports), module.name


def test_the_dialogue_layer_still_knows_nothing_about_audio_or_telephony() -> None:
    for module in _modules(DIALOGUE):
        imports = _imports(module.read_text())
        for forbidden in ("app.audio", "app.telephony", "app.api"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_the_tool_layer_still_knows_nothing_about_the_layers_above_it() -> None:
    for module in _modules(TOOLS):
        imports = _imports(module.read_text())
        for forbidden in ("app.dialogue", "app.providers", "app.audio", "app.telephony"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_the_calendar_still_knows_nothing_about_any_of_them() -> None:
    for module in _modules(CALENDAR):
        imports = _imports(module.read_text())
        for forbidden in (
            "app.dialogue",
            "app.providers",
            "app.tools",
            "app.audio",
            "app.telephony",
        ):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_the_browser_harness_is_untouched_by_telephony() -> None:
    """Two adapters, neither aware of the other.

    Imports rather than prose: the harness docstring mentions the carrier to
    say what it is *not*, and that sentence is milestone 5's to keep.
    """
    imports = _imports((APP / "api" / "harness.py").read_text())

    assert not any(name.startswith("app.telephony") for name in imports)
    assert not any(name.startswith("app.audio.telephony") for name in imports)


def test_only_one_module_still_imports_the_anthropic_sdk() -> None:
    importers = [
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if "anthropic" in _roots(module.read_text())
    ]
    assert importers == ["providers/anthropic_llm.py"]


def test_no_carrier_sdk_is_installed_or_imported() -> None:
    """Three things were needed from the carrier, and all three are stdlib."""
    for module in _modules(APP):
        assert "twilio" not in _roots(module.read_text()), module.name
