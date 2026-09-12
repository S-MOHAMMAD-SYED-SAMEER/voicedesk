"""Twilio Media Stream frames, read into something the rest of the code knows.

This is the only module in VoiceDesk that understands Twilio's JSON. Above it
there are four small dataclasses and nothing about a carrier's wire format;
below it there is audio conversion that has never heard of Twilio. Swapping
carriers touches this file and its sibling modules, and nothing else.

Twilio sends five kinds of frame the stream cares about — `connected`,
`start`, `media`, `stop` and `mark` — and others it may add at any time.
Nothing here raises for an event it does not recognise: a carrier introducing
`dtmf` must not be able to end somebody's phone call.

What *is* refused is a frame this system cannot act on safely: audio it cannot
decode, or a media format it cannot read. Those come back as
`MalformedEvent`, which the socket logs and steps over.
"""

import base64
import binascii
import json
from dataclasses import dataclass, field
from typing import Any

CONNECTED = "connected"
START = "start"
MEDIA = "media"
STOP = "stop"
MARK = "mark"
CLEAR = "clear"

# What a telephone carrier sends, and the only thing that can be decoded here.
EXPECTED_ENCODING = "audio/x-mulaw"
EXPECTED_SAMPLE_RATE = 8000
EXPECTED_CHANNELS = 1
# Only the caller's own audio. Our replies come back on the outbound track and
# feeding those to speech recognition would be a loop.
INBOUND_TRACK = "inbound"


class MalformedEvent(Exception):
    """A frame that could not be read. Logged and skipped, never fatal."""


class UnsupportedMediaFormat(Exception):
    """The carrier offered audio this system will not try to decode."""


@dataclass(frozen=True)
class ConnectedEvent:
    """The socket is up. Carries no call identity, so nothing to do yet."""

    protocol: str = ""
    version: str = ""


@dataclass(frozen=True)
class StartEvent:
    """The call is identified. This is where everything begins."""

    stream_sid: str
    call_sid: str
    account_sid: str = ""
    encoding: str = ""
    sample_rate: int = 0
    channels: int = 0
    tracks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MediaEvent:
    """20 milliseconds of somebody speaking, already base64-decoded."""

    stream_sid: str
    track: str
    audio: bytes
    chunk: int | None = None
    # Twilio's own clock, recorded for logs and otherwise unused: scheduling
    # playback from it is jitter-buffer work and belongs with latency.
    timestamp: int | None = None

    @property
    def is_inbound(self) -> bool:
        return self.track == INBOUND_TRACK


@dataclass(frozen=True)
class StopEvent:
    """The caller hung up, or the carrier ended the stream."""

    stream_sid: str
    call_sid: str = ""


@dataclass(frozen=True)
class MarkEvent:
    """A marker sent earlier has now been played to the caller.

    This is how playback completion is observed. Without it there is no way to
    know whether a reply was heard or is still sitting in the carrier's
    buffer, and therefore no way to measure a turn or to know whether an
    interruption still has something to interrupt.
    """

    stream_sid: str
    name: str = ""


@dataclass(frozen=True)
class UnknownEvent:
    """Something this milestone does not handle. Logged, then ignored."""

    name: str


Event = (
    ConnectedEvent | StartEvent | MediaEvent | StopEvent | MarkEvent | UnknownEvent
)


def parse_event(raw: str | bytes) -> Event:
    """One frame from the carrier, as a typed event."""
    try:
        body = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise MalformedEvent(f"The frame is not JSON: {exc}") from exc

    if not isinstance(body, dict):
        raise MalformedEvent("The frame is not a JSON object.")

    name = body.get("event")
    if not isinstance(name, str) or not name:
        raise MalformedEvent("The frame has no event name.")

    if name == CONNECTED:
        return ConnectedEvent(
            protocol=str(body.get("protocol", "")),
            version=str(body.get("version", "")),
        )
    if name == START:
        return _start(body)
    if name == MEDIA:
        return _media(body)
    if name == STOP:
        return _stop(body)
    if name == MARK:
        return _mark(body)
    return UnknownEvent(name=name)


def _stream_sid(body: dict[str, Any]) -> str:
    stream_sid = body.get("streamSid")
    if not isinstance(stream_sid, str) or not stream_sid:
        raise MalformedEvent(f"A {body.get('event')!r} frame has no streamSid.")
    return stream_sid


def _start(body: dict[str, Any]) -> StartEvent:
    stream_sid = _stream_sid(body)
    start = body.get("start")
    if not isinstance(start, dict):
        raise MalformedEvent("A start frame has no start object.")

    call_sid = start.get("callSid")
    if not isinstance(call_sid, str) or not call_sid:
        raise MalformedEvent("A start frame has no callSid.")

    media_format = start.get("mediaFormat")
    media_format = media_format if isinstance(media_format, dict) else {}
    tracks = start.get("tracks")

    return StartEvent(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=str(start.get("accountSid", "")),
        encoding=str(media_format.get("encoding", "")),
        sample_rate=_as_int(media_format.get("sampleRate")),
        channels=_as_int(media_format.get("channels")),
        tracks=[str(track) for track in tracks] if isinstance(tracks, list) else [],
    )


def _media(body: dict[str, Any]) -> MediaEvent:
    stream_sid = _stream_sid(body)
    media = body.get("media")
    if not isinstance(media, dict):
        raise MalformedEvent("A media frame has no media object.")

    payload = media.get("payload")
    if not isinstance(payload, str) or not payload:
        raise MalformedEvent("A media frame has no payload.")

    try:
        audio = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MalformedEvent(f"A media payload is not valid base64: {exc}") from exc

    return MediaEvent(
        stream_sid=stream_sid,
        # Twilio omits the track on single-track streams, which are inbound.
        track=str(media.get("track", INBOUND_TRACK)),
        audio=audio,
        chunk=_optional_int(media.get("chunk")),
        timestamp=_optional_int(media.get("timestamp")),
    )


def _stop(body: dict[str, Any]) -> StopEvent:
    stop = body.get("stop")
    stop = stop if isinstance(stop, dict) else {}
    return StopEvent(
        stream_sid=_stream_sid(body), call_sid=str(stop.get("callSid", ""))
    )


def _mark(body: dict[str, Any]) -> MarkEvent:
    mark = body.get("mark")
    mark = mark if isinstance(mark, dict) else {}
    return MarkEvent(stream_sid=_stream_sid(body), name=str(mark.get("name", "")))


def require_supported_format(event: StartEvent) -> None:
    """Refuse audio this system would only mangle.

    Decoding something that is not µ-law as if it were produces noise rather
    than an error, and noise reaching speech recognition looks like a caller
    who cannot be understood.
    """
    if event.encoding != EXPECTED_ENCODING:
        raise UnsupportedMediaFormat(
            f"Expected {EXPECTED_ENCODING}; the carrier offered "
            f"{event.encoding or 'nothing'}."
        )
    if event.sample_rate != EXPECTED_SAMPLE_RATE:
        raise UnsupportedMediaFormat(
            f"Expected {EXPECTED_SAMPLE_RATE} Hz; the carrier offered "
            f"{event.sample_rate or 'nothing'}."
        )
    if event.channels != EXPECTED_CHANNELS:
        raise UnsupportedMediaFormat(
            f"Expected {EXPECTED_CHANNELS} channel; the carrier offered "
            f"{event.channels or 'nothing'}."
        )


def media_frame(stream_sid: str, payload: bytes) -> dict[str, Any]:
    """One outbound frame of audio, in the shape the carrier expects."""
    return {
        "event": MEDIA,
        "streamSid": stream_sid,
        "media": {"payload": base64.b64encode(payload).decode("ascii")},
    }


def mark_frame(stream_sid: str, name: str) -> dict[str, Any]:
    """A marker the carrier echoes back when the audio before it has played."""
    return {"event": MARK, "streamSid": stream_sid, "mark": {"name": name}}


def clear_frame(stream_sid: str) -> dict[str, Any]:
    """Throw away whatever audio the carrier has buffered but not yet played.

    What makes an interruption audible. Without it a caller who interrupts
    still hears the rest of the sentence they interrupted, because the
    carrier already has it.
    """
    return {"event": CLEAR, "streamSid": stream_sid}


def _as_int(value: Any) -> int:
    return _optional_int(value) or 0


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return None
