"""Giving the log lines this application already writes somewhere to go.

There are three dozen `logger.*` calls across the application and, until now,
no configuration at all: they reached stderr through Python's last-resort
handler, without a timestamp, a level name or the module that wrote them.
This adds the configuration and nothing else. No log line was added, removed
or reworded to make it fit.

**What is deliberately not logged.** No transcript, no caller name, no
telephone number, no tool argument, no API key and no database URL appears in
any log record this application writes — that was true before this module and
`tests/test_logging.py` is what keeps it true. What does appear: call
identifiers, provider and tool names, latencies, generation numbers, and
exception text from the adapters. Enough to diagnose a call; not enough to
read one.

Two formats. `text` for a person reading a terminal, `json` for anything that
ships lines to a collector. There is no metrics endpoint, no tracing and no
Prometheus: this is a logging configuration, not an observability platform.
"""

import json
import logging
import logging.config
from typing import Any

from app.config import Settings, get_settings

# Attributes `logging` puts on every record. Anything else a caller passed
# through `extra=` is ours, and goes into the JSON object.
_BUILT_IN = frozenset(
    """
    args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName
    relativeCreated stack_info thread threadName taskName
    """.split()
)

TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with whatever context the call site added."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _BUILT_IN and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(settings: Settings | None = None) -> None:
    """Install the application's logging configuration.

    Called once from the application lifespan. `disable_existing_loggers` is
    false on purpose: Uvicorn's own loggers are already set up by the time
    this runs, and silencing them would lose the access log.
    """
    resolved = settings or get_settings()

    formatter = (
        {"()": f"{__name__}.JsonFormatter"}
        if resolved.log_format == "json"
        else {"format": TEXT_FORMAT}
    )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"voicedesk": formatter},
            "handlers": {
                "stderr": {
                    "class": "logging.StreamHandler",
                    "formatter": "voicedesk",
                    "stream": "ext://sys.stderr",
                }
            },
            # Only this application's loggers are configured. Uvicorn and
            # SQLAlchemy keep whatever the server gave them, so turning our
            # level down does not silence the access log.
            "loggers": {
                "app": {
                    "handlers": ["stderr"],
                    "level": resolved.log_level,
                    "propagate": False,
                }
            },
        }
    )


def call_context(call_id: object = None, call_sid: object = None) -> dict[str, Any]:
    """The identifiers a call-scoped log line should carry.

    Identifiers only. A call id is a UUID this system minted and a call SID is
    the carrier's own reference — neither says anything about who was on the
    telephone, which is the point of correlating on them rather than on a
    number.
    """
    context: dict[str, Any] = {}
    if call_id is not None:
        context["call_id"] = str(call_id)
    if call_sid:
        context["call_sid"] = str(call_sid)
    return context


__all__ = ["TEXT_FORMAT", "JsonFormatter", "call_context", "configure"]
