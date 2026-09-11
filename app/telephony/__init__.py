"""Telephony: a real phone call joining the same dialogue as the harness.

    caller → carrier → POST /telephony/voice → TwiML → WS /telephony/stream
                                                            ↓
                                      µ-law ⇄ PCM  (app/audio/telephony.py)
                                                            ↓
                                   VoiceSession → Conversation → tools

Two modules face the carrier: `webhook.py` answers the call and `stream.py`
carries it. `events.py` is the only place that knows the carrier's JSON, and
the audio conversion below it has never heard of a carrier at all. Nothing
here contains dialogue logic, reaches the calendar, runs a tool or speaks to a
model — the path to all of that is `VoiceSession`, exactly as it is for the
browser.

Everything is off unless `telephony_enabled` is set.
"""

from fastapi import APIRouter

from app.telephony import stream, webhook
from app.telephony.events import (
    ConnectedEvent,
    MalformedEvent,
    MediaEvent,
    StartEvent,
    StopEvent,
    UnknownEvent,
    UnsupportedMediaFormat,
    parse_event,
)

router = APIRouter()
router.include_router(webhook.router)
router.include_router(stream.router)

__all__ = [
    "ConnectedEvent",
    "MalformedEvent",
    "MediaEvent",
    "StartEvent",
    "StopEvent",
    "UnknownEvent",
    "UnsupportedMediaFormat",
    "parse_event",
    "router",
]
