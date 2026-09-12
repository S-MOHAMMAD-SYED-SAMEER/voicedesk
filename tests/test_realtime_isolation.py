"""The boundaries milestone 7 is not allowed to cross.

    transport → realtime → dialogue → tools → calendar → database

and never realtime → calendar, realtime → a tool, or realtime → a model.
Parsed from source, so a docstring may name a thing a module must not import.
"""

import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parent.parent / "app"
REALTIME = APP / "realtime"
AUDIO = APP / "audio"
DIALOGUE = APP / "dialogue"
TOOLS = APP / "tools"
CALENDAR = APP / "calendar"
TELEPHONY = APP / "telephony"
VAD = AUDIO / "vad.py"
ENDPOINT = AUDIO / "endpoint.py"

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


# --- the realtime layer stays an adapter ----------------------------------


def test_no_realtime_module_reaches_the_calendar() -> None:
    for module in _modules(REALTIME):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.calendar") for name in imports), module.name


def test_no_realtime_module_reaches_the_tools() -> None:
    """It has no business booking anything. That is what the dialogue is for."""
    for module in _modules(REALTIME):
        imports = _imports(module.read_text())
        assert not any(name.startswith("app.tools") for name in imports), module.name


def test_no_realtime_module_imports_a_vendor_sdk() -> None:
    for module in _modules(REALTIME):
        assert not _roots(module.read_text()) & VENDOR_SDKS, module.name


def test_no_realtime_module_imports_a_provider_implementation() -> None:
    """It is handed providers; it does not choose them."""
    for module in _modules(REALTIME):
        imports = _imports(module.read_text())
        for forbidden in (
            "app.providers.anthropic_llm",
            "app.providers.deepgram_stream_stt",
            "app.providers.elevenlabs_stream_tts",
            "app.providers.factory",
        ):
            assert forbidden not in imports, f"{module.name} imports {forbidden}"


def test_realtime_reaches_the_dialogue_only_through_conversation() -> None:
    """One dialogue path. Press-to-talk takes the same one."""
    imported = set()
    for module in _modules(REALTIME):
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "app.dialogue":
                imported.update(alias.name for alias in node.names)
    assert imported == {"Conversation", "DialogueResult"}


def test_realtime_knows_nothing_about_any_transport() -> None:
    for module in _modules(REALTIME):
        imports = _imports(module.read_text())
        for forbidden in ("app.telephony", "app.api", "fastapi", "starlette"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_only_the_latency_writer_touches_a_model() -> None:
    """Everything else in the layer is about audio, not rows."""
    touching = {
        module.name
        for module in _modules(REALTIME)
        if any(name.startswith("app.models") for name in _imports(module.read_text()))
    }
    assert touching == {"latency.py"}


def test_the_latency_writer_touches_only_the_turn_model() -> None:
    imported = set()
    for node in ast.walk(ast.parse((REALTIME / "latency.py").read_text())):
        if isinstance(node, ast.ImportFrom) and node.module == "app.models":
            imported.update(alias.name for alias in node.names)
    assert imported == {"Turn"}


# --- voice activity knows nothing about anything --------------------------


def test_the_detector_is_pure_signal_processing() -> None:
    """No transport, no provider, no dialogue, no configuration."""
    imports = _imports(VAD.read_text())

    assert not any(name.startswith("app.") for name in imports)
    assert _roots(VAD.read_text()) <= {"math", "struct", "dataclasses"}


def test_the_endpointer_knows_only_the_detector() -> None:
    imports = _imports(ENDPOINT.read_text())

    assert {name for name in imports if name.startswith("app.")} == {"app.audio.vad"}


def test_neither_knows_about_a_carrier() -> None:
    for module in (VAD, ENDPOINT):
        source = module.read_text().lower()
        assert "twilio" not in source
        assert "streamsid" not in source


# --- the streaming interfaces stay vendor-neutral -------------------------


def test_the_streaming_interfaces_name_no_vendor() -> None:
    for name in ("streaming_stt.py", "streaming_tts.py"):
        source = (APP / "providers" / name).read_text().lower()
        for vendor in ("deepgram", "elevenlabs", "anthropic", "twilio"):
            assert vendor not in source, f"{name} names {vendor}"


def test_the_streaming_interfaces_import_no_vendor_sdk() -> None:
    for name in ("streaming_stt.py", "streaming_tts.py"):
        assert not _roots((APP / "providers" / name).read_text()) & VENDOR_SDKS


def test_the_offline_streaming_providers_reach_no_network() -> None:
    source = (APP / "providers" / "offline_streaming.py").read_text()

    assert not _roots(source) & {"httpx", "requests", "urllib", "socket", "websockets"}


def test_each_streaming_adapter_is_the_only_one_that_knows_its_service() -> None:
    owners = {
        "api.deepgram.com": {
            "providers/deepgram_stt.py",
            "providers/deepgram_stream_stt.py",
        },
        "api.elevenlabs.io": {
            "providers/elevenlabs_tts.py",
            "providers/elevenlabs_stream_tts.py",
        },
    }
    for marker, allowed in owners.items():
        knows = {
            module.relative_to(APP).as_posix()
            for module in _modules(APP)
            if marker in module.read_text()
        }
        assert knows <= allowed, f"{marker!r} also appears in {knows - allowed}"


# --- the layers below are unchanged ---------------------------------------


def test_the_dialogue_layer_still_knows_nothing_about_the_layers_above_it() -> None:
    for module in _modules(DIALOGUE):
        imports = _imports(module.read_text())
        for forbidden in ("app.audio", "app.realtime", "app.telephony", "app.api"):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_the_tool_layer_still_knows_nothing_about_the_layers_above_it() -> None:
    for module in _modules(TOOLS):
        imports = _imports(module.read_text())
        for forbidden in (
            "app.dialogue",
            "app.providers",
            "app.audio",
            "app.realtime",
            "app.telephony",
        ):
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
            "app.realtime",
            "app.telephony",
        ):
            assert not any(name.startswith(forbidden) for name in imports), (
                f"{module.name} imports {forbidden}"
            )


def test_the_carrier_layer_still_knows_the_json_and_nothing_else_does() -> None:
    for marker in ("streamSid", "callSid", "mediaFormat"):
        knows = {
            module.relative_to(APP).as_posix()
            for module in _modules(APP)
            if marker in module.read_text()
        }
        assert knows == {"telephony/events.py"}, f"{marker!r} appears in {knows}"


def test_the_audio_conversion_still_knows_nothing_about_the_carrier() -> None:
    source = (AUDIO / "telephony.py").read_text()

    assert "twilio" not in source.lower()
    assert not any(name.startswith("app.telephony") for name in _imports(source))


def test_the_press_to_talk_session_is_untouched_by_realtime() -> None:
    """Two sessions, neither aware of the other, one dialogue beneath both."""
    imports = _imports((AUDIO / "session.py").read_text())

    assert not any(name.startswith("app.realtime") for name in imports)


def test_only_one_module_still_imports_the_anthropic_sdk() -> None:
    importers = [
        module.relative_to(APP).as_posix()
        for module in _modules(APP)
        if "anthropic" in _roots(module.read_text())
    ]
    assert importers == ["providers/anthropic_llm.py"]


def test_no_carrier_sdk_is_imported_anywhere() -> None:
    for module in _modules(APP):
        assert "twilio" not in _roots(module.read_text()), module.name


def test_no_vad_or_dsp_dependency_was_added() -> None:
    """The detector is standard library, and the audit says so."""
    forbidden = {"numpy", "torch", "onnxruntime", "webrtcvad", "scipy", "silero_vad"}
    for module in _modules(APP):
        assert not _roots(module.read_text()) & forbidden, module.name
