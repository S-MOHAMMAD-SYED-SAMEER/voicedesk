"""The instruction sent back to the carrier, and proving the carrier sent it."""

import pytest

from app.telephony.security import (
    SIGNATURE_HEADER,
    expected_signature,
    is_valid_signature,
)
from app.telephony.twiml import STREAM_PATH, TwiMLError, connect_stream, stream_url

TOKEN = "an-auth-token"
URL = "https://voicedesk.example.com/telephony/voice"
PARAMS = {"CallSid": "CA1", "From": "+447700900123", "To": "+441234567890"}


# --- TwiML ----------------------------------------------------------------


def test_the_instruction_connects_a_media_stream() -> None:
    body = connect_stream("https://voicedesk.example.com")

    assert "<Connect>" in body
    assert "<Stream url=" in body
    assert body.startswith('<?xml version="1.0" encoding="UTF-8"?><Response>')
    assert body.endswith("</Connect></Response>")


def test_the_instruction_never_speaks() -> None:
    """The receptionist has a voice already; a second one would be a stranger."""
    assert "<Say" not in connect_stream("https://voicedesk.example.com")


def test_the_stream_address_is_a_secure_websocket() -> None:
    assert (
        stream_url("https://voicedesk.example.com")
        == f"wss://voicedesk.example.com{STREAM_PATH}"
    )


def test_a_plain_http_base_gives_a_plain_websocket() -> None:
    """For a local tunnel; the carrier will not accept it from the internet."""
    assert stream_url("http://localhost:8000") == f"ws://localhost:8000{STREAM_PATH}"


def test_a_bare_host_is_assumed_secure() -> None:
    assert stream_url("voicedesk.example.com") == f"wss://voicedesk.example.com{STREAM_PATH}"


def test_a_trailing_slash_does_not_double_up() -> None:
    assert stream_url("https://voicedesk.example.com/") == (
        f"wss://voicedesk.example.com{STREAM_PATH}"
    )


def test_a_base_path_is_kept() -> None:
    assert stream_url("https://example.com/voicedesk") == (
        f"wss://example.com/voicedesk{STREAM_PATH}"
    )


def test_the_address_is_escaped_into_the_attribute() -> None:
    """An ampersand in the base would otherwise produce something that is not XML."""
    body = connect_stream("https://example.com/a&b")

    assert "&amp;b" in body
    assert "url=\"wss://example.com/a&amp;b/telephony/stream\"" in body


@pytest.mark.parametrize("base", ["", "   ", None])
def test_no_configured_address_is_refused_rather_than_guessed(base) -> None:
    """Behind a proxy the request describes the hop, not the public address."""
    with pytest.raises(TwiMLError, match="public base URL"):
        connect_stream(base or "")


# --- signatures -----------------------------------------------------------


def test_a_correctly_signed_request_is_accepted() -> None:
    signature = expected_signature(TOKEN, URL, PARAMS)

    assert is_valid_signature(TOKEN, URL, PARAMS, signature)


def test_the_signature_is_stable() -> None:
    assert expected_signature(TOKEN, URL, PARAMS) == expected_signature(
        TOKEN, URL, PARAMS
    )


def test_parameter_order_does_not_change_the_signature() -> None:
    """The algorithm sorts by key, so a re-ordered form is the same request."""
    reversed_params = dict(reversed(list(PARAMS.items())))

    assert expected_signature(TOKEN, URL, reversed_params) == expected_signature(
        TOKEN, URL, PARAMS
    )


def test_a_different_token_is_rejected() -> None:
    signature = expected_signature(TOKEN, URL, PARAMS)

    assert not is_valid_signature("another-token", URL, PARAMS, signature)


def test_an_altered_parameter_is_rejected() -> None:
    signature = expected_signature(TOKEN, URL, PARAMS)
    tampered = PARAMS | {"From": "+440000000000"}

    assert not is_valid_signature(TOKEN, URL, tampered, signature)


def test_an_added_parameter_is_rejected() -> None:
    signature = expected_signature(TOKEN, URL, PARAMS)

    assert not is_valid_signature(TOKEN, URL, PARAMS | {"Extra": "1"}, signature)


def test_an_altered_url_is_rejected() -> None:
    signature = expected_signature(TOKEN, URL, PARAMS)

    assert not is_valid_signature(TOKEN, "https://evil.example.com/voice", PARAMS, signature)


@pytest.mark.parametrize("signature", [None, "", "not-a-signature"])
def test_a_missing_or_nonsense_signature_is_rejected(signature) -> None:
    assert not is_valid_signature(TOKEN, URL, PARAMS, signature)


def test_an_unconfigured_token_rejects_everything() -> None:
    """A deployment mistake must not become an open endpoint."""
    assert not is_valid_signature("", URL, PARAMS, expected_signature("", URL, PARAMS))


def test_a_request_with_no_parameters_can_still_be_signed() -> None:
    signature = expected_signature(TOKEN, URL, {})

    assert is_valid_signature(TOKEN, URL, {}, signature)


def test_the_header_is_the_documented_one() -> None:
    assert SIGNATURE_HEADER == "X-Twilio-Signature"


def test_the_secret_never_appears_in_the_signature() -> None:
    assert TOKEN not in expected_signature(TOKEN, URL, PARAMS)
