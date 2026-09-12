"""Log lines go somewhere, in one of two shapes, and carry nothing private.

The second half is the one that matters. This application holds transcripts,
caller names and telephone numbers, and it holds four provider keys and a
database password. None of that has ever appeared in a log line — these tests
are what keeps it that way when somebody adds the next `logger.info`.
"""

import ast
import io
import json
import logging
import pathlib

import pytest

from app.config import Settings
from app.logging import TEXT_FORMAT, JsonFormatter, call_context, configure

APP = pathlib.Path(__file__).resolve().parent.parent / "app"


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.telephony.stream",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Call %s started",
        args=("CA1",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# --- the JSON shape --------------------------------------------------------


def test_a_record_becomes_one_json_object() -> None:
    payload = json.loads(JsonFormatter().format(_record()))

    assert payload["message"] == "Call CA1 started"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.telephony.stream"
    assert payload["time"]


def test_json_is_one_line() -> None:
    """Collectors read lines. A pretty-printed object is several."""
    assert "\n" not in JsonFormatter().format(_record())


def test_extra_context_reaches_the_object() -> None:
    payload = json.loads(JsonFormatter().format(_record(call_id="abc", latency_ms=42)))

    assert payload["call_id"] == "abc"
    assert payload["latency_ms"] == 42


def test_the_built_in_attributes_stay_out_of_it() -> None:
    """Otherwise every line carries a filename, a thread id and a process id."""
    payload = json.loads(JsonFormatter().format(_record()))

    assert "pathname" not in payload
    assert "thread" not in payload
    assert "msg" not in payload


def test_an_exception_is_rendered_into_the_object() -> None:
    try:
        raise ValueError("the provider fell over")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError" in payload["exception"]


def test_something_unserialisable_does_not_lose_the_line() -> None:
    """A log call is not worth an exception inside the logger."""
    payload = json.loads(JsonFormatter().format(_record(thing=object())))

    assert "object at 0x" in payload["thing"]


# --- configuring -----------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_logging():
    """Put the application logger back however the suite found it."""
    logger = logging.getLogger("app")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    try:
        yield
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def _captured(settings: Settings, message: str = "hello", **extra) -> str:
    configure(settings)
    logger = logging.getLogger("app.test")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.getLogger("app").handlers[0].formatter)
    logging.getLogger("app").handlers = [handler]
    logger.info(message, extra=extra)
    return stream.getvalue()


def test_text_is_the_default_shape() -> None:
    written = _captured(Settings(_env_file=None))

    assert "INFO" in written
    assert "app.test" in written
    assert not written.startswith("{")


def test_json_can_be_asked_for() -> None:
    written = _captured(Settings(_env_file=None, log_format="json"))

    assert json.loads(written)["message"] == "hello"


def test_the_level_comes_from_configuration() -> None:
    configure(Settings(_env_file=None, log_level="WARNING"))

    assert logging.getLogger("app").level == logging.WARNING


def test_the_level_is_case_insensitive() -> None:
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"


def test_a_level_that_is_not_one_is_refused() -> None:
    with pytest.raises(ValueError, match="not a log level"):
        Settings(_env_file=None, log_level="chatty")


def test_a_format_that_is_not_one_is_refused() -> None:
    with pytest.raises(ValueError, match="not a log format"):
        Settings(_env_file=None, log_format="xml")


def test_only_this_application_is_configured() -> None:
    """Uvicorn keeps its own handlers, so the access log survives."""
    configure(Settings(_env_file=None))

    assert logging.getLogger("uvicorn.access").handlers != logging.getLogger(
        "app"
    ).handlers


def test_existing_loggers_are_not_disabled() -> None:
    other = logging.getLogger("somebody.else")
    configure(Settings(_env_file=None))

    assert other.disabled is False


def test_the_text_format_names_the_logger_and_the_level() -> None:
    assert "%(levelname)" in TEXT_FORMAT
    assert "%(name)s" in TEXT_FORMAT


# --- call context ----------------------------------------------------------


def test_call_context_is_identifiers_only() -> None:
    context = call_context(call_id="a4f", call_sid="CA1")

    assert context == {"call_id": "a4f", "call_sid": "CA1"}


def test_call_context_leaves_out_what_it_was_not_given() -> None:
    assert call_context(call_sid="CA1") == {"call_sid": "CA1"}
    assert call_context() == {}


# --- what must never be logged ---------------------------------------------

SECRETS = (
    "api_key",
    "auth_token",
    "database_url",
    "eval_database_url",
    "password",
)

PRIVATE = (
    "customer_name",
    "caller_name",
    "from_number",
    "to_number",
    "phone",
    "transcript",
)


def _log_arguments(path: pathlib.Path) -> list[ast.expr]:
    """Every argument handed to a `logger.*` call in one module."""
    found: list[ast.expr] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        if not isinstance(function.value, ast.Name) or function.value.id != "logger":
            continue
        if function.attr not in {
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "critical",
        }:
            continue
        found.extend(node.args)
        found.extend(keyword.value for keyword in node.keywords)
    return found


def _sources() -> list[pathlib.Path]:
    return sorted(APP.rglob("*.py"))


def _names(expression: ast.expr) -> set[str]:
    """Every identifier the expression mentions, however deeply."""
    found = set()
    for node in ast.walk(expression):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


@pytest.mark.parametrize("forbidden", SECRETS)
def test_no_log_line_is_handed_a_secret(forbidden: str) -> None:
    """Not the value, and not the attribute that holds it."""
    for path in _sources():
        for argument in _log_arguments(path):
            assert forbidden not in _names(argument), f"{path.name}: {forbidden}"


@pytest.mark.parametrize("forbidden", PRIVATE)
def test_no_log_line_is_handed_something_about_the_caller(forbidden: str) -> None:
    """Identifiers correlate a call. Names and numbers identify a person."""
    for path in _sources():
        for argument in _log_arguments(path):
            assert forbidden not in _names(argument), f"{path.name}: {forbidden}"


def test_nothing_prints_to_standard_output_outside_the_commands() -> None:
    """`print` in a request path writes past every handler and every level.

    The three command-line entry points are the exception: a report a person
    asked for on a terminal belongs on standard output, not in a log.
    """
    allowed = {"evals", "metrics", "retention"}

    for path in _sources():
        if path.stem in allowed or path.parent.name in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                raise AssertionError(f"{path.relative_to(APP)} prints")
