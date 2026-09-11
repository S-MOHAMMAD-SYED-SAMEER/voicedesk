"""Reading a carrier's frames, including the ones that make no sense."""

import base64
import json

import pytest

from app.telephony.events import (
    ConnectedEvent,
    MalformedEvent,
    MediaEvent,
    StartEvent,
    StopEvent,
    UnknownEvent,
    UnsupportedMediaFormat,
    media_frame,
    mark_frame,
    parse_event,
    require_supported_format,
)

STREAM = "MZ00000000000000000000000000000000"
CALL = "CA00000000000000000000000000000000"


def _start(**overrides) -> str:
    media_format = {
        "encoding": "audio/x-mulaw",
        "sampleRate": 8000,
        "channels": 1,
    }
    media_format.update(overrides.pop("mediaFormat", {}))
    start = {
        "callSid": CALL,
        "accountSid": "AC1",
        "tracks": ["inbound"],
        "mediaFormat": media_format,
    }
    start.update(overrides.pop("start", {}))
    body = {"event": "start", "streamSid": STREAM, "start": start}
    body.update(overrides)
    return json.dumps(body)


def _media(payload: bytes = b"\xff\xff", **overrides) -> str:
    media = {
        "track": "inbound",
        "chunk": 3,
        "timestamp": 60,
        "payload": base64.b64encode(payload).decode(),
    }
    media.update(overrides.pop("media", {}))
    body = {"event": "media", "streamSid": STREAM, "media": media}
    body.update(overrides)
    return json.dumps(body)


# --- the four events that matter ------------------------------------------


def test_connected_is_read() -> None:
    event = parse_event(
        json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})
    )

    assert isinstance(event, ConnectedEvent)
    assert (event.protocol, event.version) == ("Call", "1.0.0")


def test_start_carries_the_call_identity_and_the_format() -> None:
    event = parse_event(_start())

    assert isinstance(event, StartEvent)
    assert (event.stream_sid, event.call_sid) == (STREAM, CALL)
    assert (event.encoding, event.sample_rate, event.channels) == (
        "audio/x-mulaw",
        8000,
        1,
    )
    assert event.tracks == ["inbound"]


def test_media_arrives_already_decoded() -> None:
    event = parse_event(_media(b"\x01\x02\x03"))

    assert isinstance(event, MediaEvent)
    assert event.audio == b"\x01\x02\x03"
    assert event.is_inbound
    assert (event.chunk, event.timestamp) == (3, 60)


def test_stop_is_read() -> None:
    event = parse_event(
        json.dumps({"event": "stop", "streamSid": STREAM, "stop": {"callSid": CALL}})
    )

    assert isinstance(event, StopEvent)
    assert (event.stream_sid, event.call_sid) == (STREAM, CALL)


def test_stop_without_a_stop_object_is_still_a_stop() -> None:
    event = parse_event(json.dumps({"event": "stop", "streamSid": STREAM}))

    assert isinstance(event, StopEvent)
    assert event.call_sid == ""


# --- events this milestone does not handle --------------------------------


@pytest.mark.parametrize("name", ["mark", "dtmf", "something-invented-later"])
def test_an_unhandled_event_is_reported_not_raised(name: str) -> None:
    """A carrier adding an event must not be able to end somebody's call."""
    event = parse_event(json.dumps({"event": name, "streamSid": STREAM}))

    assert isinstance(event, UnknownEvent)
    assert event.name == name


# --- frames that cannot be read -------------------------------------------


@pytest.mark.parametrize(
    "raw", ["", "not json", "{", "[1, 2]", '"a string"', "null"],
    ids=["empty", "text", "truncated", "array", "string", "null"],
)
def test_a_frame_that_is_not_a_json_object_is_malformed(raw: str) -> None:
    with pytest.raises(MalformedEvent):
        parse_event(raw)


@pytest.mark.parametrize("body", [{}, {"event": ""}, {"event": 7}, {"streamSid": STREAM}])
def test_a_frame_with_no_event_name_is_malformed(body: dict) -> None:
    with pytest.raises(MalformedEvent, match="no event name"):
        parse_event(json.dumps(body))


@pytest.mark.parametrize("name", ["start", "media", "stop"])
def test_a_frame_with_no_stream_is_malformed(name: str) -> None:
    with pytest.raises(MalformedEvent, match="streamSid"):
        parse_event(json.dumps({"event": name}))


def test_a_start_with_no_start_object_is_malformed() -> None:
    with pytest.raises(MalformedEvent, match="no start object"):
        parse_event(json.dumps({"event": "start", "streamSid": STREAM}))


def test_a_start_with_no_call_is_malformed() -> None:
    with pytest.raises(MalformedEvent, match="callSid"):
        parse_event(_start(start={"callSid": ""}))


def test_media_with_no_media_object_is_malformed() -> None:
    with pytest.raises(MalformedEvent, match="no media object"):
        parse_event(json.dumps({"event": "media", "streamSid": STREAM}))


def test_media_with_no_payload_is_malformed() -> None:
    with pytest.raises(MalformedEvent, match="no payload"):
        parse_event(_media(media={"payload": ""}))


def test_a_payload_that_is_not_base64_is_malformed() -> None:
    """Decoding it loosely would hand speech recognition noise."""
    with pytest.raises(MalformedEvent, match="base64"):
        parse_event(_media(media={"payload": "not base64!!"}))


# --- the media format -----------------------------------------------------


def test_the_carriers_own_format_is_accepted() -> None:
    require_supported_format(parse_event(_start()))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("encoding", "audio/l16", "audio/x-mulaw"),
        ("sampleRate", 16000, "8000 Hz"),
        ("channels", 2, "1 channel"),
    ],
)
def test_anything_else_is_refused_rather_than_decoded(field, value, message) -> None:
    """Decoding non-µ-law as µ-law produces noise, not an error."""
    event = parse_event(_start(mediaFormat={field: value}))

    with pytest.raises(UnsupportedMediaFormat, match=message):
        require_supported_format(event)


def test_a_missing_format_is_refused() -> None:
    event = parse_event(_start(start={"mediaFormat": {}}))

    with pytest.raises(UnsupportedMediaFormat):
        require_supported_format(event)


# --- tracks ---------------------------------------------------------------


def test_the_outbound_track_is_marked_as_not_inbound() -> None:
    """It is our own voice coming back; transcribing it would be a loop."""
    event = parse_event(_media(media={"track": "outbound"}))

    assert not event.is_inbound


def test_a_frame_with_no_track_is_treated_as_the_callers() -> None:
    """Single-track streams omit it, and a single track is the caller."""
    body = json.dumps(
        {
            "event": "media",
            "streamSid": STREAM,
            "media": {"payload": base64.b64encode(b"\xff").decode()},
        }
    )

    event = parse_event(body)

    assert event.track == "inbound"
    assert event.is_inbound


# --- numbers that arrive as strings ---------------------------------------


def test_numeric_fields_sent_as_strings_are_read() -> None:
    event = parse_event(_start(mediaFormat={"sampleRate": "8000", "channels": "1"}))

    assert (event.sample_rate, event.channels) == (8000, 1)
    require_supported_format(event)


def test_unreadable_numbers_become_zero_and_are_refused() -> None:
    event = parse_event(_start(mediaFormat={"sampleRate": "eight thousand"}))

    assert event.sample_rate == 0
    with pytest.raises(UnsupportedMediaFormat):
        require_supported_format(event)


# --- outbound frames ------------------------------------------------------


def test_an_outbound_media_frame_is_shaped_as_the_carrier_expects() -> None:
    frame = media_frame(STREAM, b"\xff\xff")

    assert frame == {
        "event": "media",
        "streamSid": STREAM,
        "media": {"payload": base64.b64encode(b"\xff\xff").decode()},
    }


def test_an_outbound_frame_round_trips_through_the_parser() -> None:
    event = parse_event(json.dumps(media_frame(STREAM, b"\x01\x02")))

    assert isinstance(event, MediaEvent)
    assert event.audio == b"\x01\x02"


def test_a_mark_names_the_end_of_a_batch_of_audio() -> None:
    assert mark_frame(STREAM, "reply-1") == {
        "event": "mark",
        "streamSid": STREAM,
        "mark": {"name": "reply-1"},
    }
