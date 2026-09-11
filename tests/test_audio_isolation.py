"""The boundaries milestone 5 is not allowed to cross.

Parsed from source rather than grepped for, so a docstring may name the thing
a module must not import without tripping the check.

The layering the whole project rests on:

    audio → dialogue → tools → calendar → database

and never audio → calendar, audio → a model, or audio → a tool.
"""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parent.parent / "app"
AUDIO = APP / "audio"
PROVIDERS = APP / "providers"
DIALOGUE = APP / "dialogue"
TOOLS = APP / "tools"
CALENDAR = APP / "calendar"
HARNESS = APP / "api" / "harness.py"

VENDOR_SDKS = {"anthropic", "openai", "google", "cohere", "mistralai"}
TELEPHONY = {"twilio", "twilio_http_client"}


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


# --- the audio layer stays an adapter --------------------------------------


def test_the_audio_layer_never_reaches_the_calendar() -> None:
    for module in _modules(AUDIO):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.calendar") for name in imports), module.name


def test_the_audio_layer_never_reaches_the_tools() -> None:
    """It has no business booking anything. That is what the dialogue is for."""
    for module in _modules(AUDIO):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.tools") for name in imports), module.name


def test_the_audio_layer_never_speaks_to_a_model() -> None:
    for module in _modules(AUDIO):
        source = module.read_text()
        assert not _roots(source) & VENDOR_SDKS, module.name
        assert "app.providers.anthropic_llm" not in _imports(source), module.name


def test_the_audio_layer_touches_no_database_model() -> None:
    """The dialogue layer owns the transcript. This layer writes nothing."""
    for module in _modules(AUDIO):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.models") for name in imports), module.name


def test_the_audio_layer_reaches_the_dialogue_only_through_conversation() -> None:
    imported = set()
    for module in _modules(AUDIO):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "app.dialogue":
                imported.update(alias.name for alias in node.names)
    assert imported == {"Conversation", "DialogueResult"}


def test_the_dialogue_layer_still_knows_nothing_about_audio() -> None:
    """The dependency arrow points one way, or the layering is a fiction."""
    for module in _modules(DIALOGUE):
        imports = _imports(module.read_text())
        for forbidden in ("app.audio", "app.api", "app.providers.stt",
                          "app.providers.tts", "app.providers.speech"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


# --- the harness endpoint ---------------------------------------------------


def test_the_harness_owns_no_dialogue_calendar_or_tool_logic() -> None:
    imports = _imports(HARNESS.read_text())
    for forbidden in ("app.calendar", "app.tools"):
        assert not any(name.startswith(forbidden) for name in imports), forbidden


def test_the_harness_imports_no_vendor_sdk() -> None:
    source = HARNESS.read_text()
    assert not _roots(source) & VENDOR_SDKS
    assert "httpx" not in _roots(source)


def test_the_harness_selects_providers_through_the_factory() -> None:
    """So one place decides which implementation anything gets."""
    imports = _imports(HARNESS.read_text())
    assert "app.providers.factory" in imports
    assert "app.providers.anthropic_llm" not in imports
    assert "app.providers.deepgram_stt" not in imports
    assert "app.providers.elevenlabs_tts" not in imports


# --- one vendor, one module --------------------------------------------------


def _importers(name: str) -> list[str]:
    return [
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if name in _roots(module.read_text())
    ]


def test_only_one_module_imports_the_anthropic_sdk() -> None:
    assert _importers("anthropic") == ["providers/anthropic_llm.py"]


def test_only_the_two_rest_adapters_use_an_http_client() -> None:
    """Everything else in the application talks to a provider interface."""
    assert _importers("httpx") == [
        "providers/deepgram_stt.py",
        "providers/elevenlabs_tts.py",
    ]


# A vendor's endpoint and its header names are the wire format. Wherever one
# of these strings appears, that module knows how to talk to that vendor — and
# exactly one module is allowed to.
WIRE_FORMAT = {
    "providers/deepgram_stt.py": ("api.deepgram.com", "Token "),
    "providers/elevenlabs_tts.py": ("api.elevenlabs.io", "xi-api-key"),
}


def test_only_one_module_knows_each_vendor_wire_format() -> None:
    for owner, markers in WIRE_FORMAT.items():
        for marker in markers:
            knows = [
                module.relative_to(APP).as_posix()
                for module in _modules(APP)
                if marker in module.read_text()
            ]
            assert knows == [owner], f"{marker!r} appears in {knows}"


def test_a_vendor_name_never_reaches_the_audio_or_dialogue_layers() -> None:
    """Selecting a provider is configuration; naming one is coupling."""
    for module in _modules(AUDIO) + _modules(DIALOGUE) + [HARNESS]:
        source = module.read_text().lower()
        for vendor in ("deepgram", "elevenlabs", "anthropic"):
            assert vendor not in source, f"{module.name} names {vendor}"


def test_the_vendor_neutral_speech_interfaces_name_no_speech_vendor() -> None:
    for module in (PROVIDERS / "stt.py", PROVIDERS / "tts.py", PROVIDERS / "speech.py"):
        source = module.read_text().lower()
        for vendor in ("deepgram", "elevenlabs", "anthropic"):
            assert vendor not in source, f"{module.name} names {vendor}"


def test_importing_the_providers_package_pulls_in_no_vendor() -> None:
    """A key-free clone must be able to import this without an SDK loading."""
    source = (PROVIDERS / "__init__.py").read_text()
    imports = _imports(source)
    for implementation in ("anthropic_llm", "deepgram_stt", "elevenlabs_tts"):
        assert f"app.providers.{implementation}" not in imports


def test_the_factory_imports_its_implementations_lazily() -> None:
    """Choosing the offline provider must not load an adapter it will not use."""
    tree = ast.parse((PROVIDERS / "factory.py").read_text())
    top_level = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for implementation in ("anthropic_llm", "deepgram_stt", "elevenlabs_tts"):
        assert f"app.providers.{implementation}" not in top_level


def test_the_offline_providers_reach_no_network() -> None:
    """They exist so a clone with no account can run the whole harness."""
    for module in (PROVIDERS / "offline_stt.py", PROVIDERS / "offline_tts.py"):
        roots = _roots(module.read_text())
        assert not roots & {"httpx", "requests", "urllib", "socket", "http"}


# --- nothing telephonic yet --------------------------------------------------


def test_no_telephony_anywhere_in_the_application() -> None:
    """Twilio, media streams and phone numbers are the next milestone's."""
    for module in _modules(APP):
        assert not _roots(module.read_text()) & TELEPHONY, module.name


def test_the_tool_layer_still_knows_nothing_about_the_layers_above_it() -> None:
    for module in _modules(TOOLS):
        imports = _imports(module.read_text())
        for forbidden in ("app.dialogue", "app.providers", "app.audio", "app.api"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_the_calendar_still_knows_nothing_about_any_of_them() -> None:
    for module in _modules(CALENDAR):
        imports = _imports(module.read_text())
        for forbidden in ("app.dialogue", "app.providers", "app.tools", "app.audio"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )
