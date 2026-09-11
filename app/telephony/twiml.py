"""The one instruction a carrier needs: open a media stream back to us.

Deliberately the smallest TwiML that works. In particular there is no `<Say>`:
the receptionist has a voice already, chosen in configuration and used by the
browser harness, and letting the carrier speak the greeting would put a second
unrelated voice on the line before the first one.
"""

from urllib.parse import urlsplit, urlunsplit
from xml.sax.saxutils import quoteattr

STREAM_PATH = "/telephony/stream"


class TwiMLError(Exception):
    """The instruction could not be built."""


def stream_url(public_base_url: str, path: str = STREAM_PATH) -> str:
    """The `wss://` address the carrier should connect its media stream to.

    Built from configuration rather than the incoming request: behind a proxy
    the request's own host and scheme describe the hop, not the address a
    carrier out on the internet has to dial back.
    """
    base = (public_base_url or "").strip().rstrip("/")
    if not base:
        raise TwiMLError(
            "No public base URL is configured, so there is no address for the "
            "carrier to stream to."
        )

    parts = urlsplit(base if "//" in base else f"//{base}", scheme="https")
    if not parts.netloc:
        raise TwiMLError(f"{public_base_url!r} is not a usable base URL.")

    scheme = {"http": "ws", "ws": "ws"}.get(parts.scheme, "wss")
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/") + path, "", ""))


def connect_stream(public_base_url: str, path: str = STREAM_PATH) -> str:
    """TwiML telling the carrier to connect a bidirectional media stream.

    The URL goes through `quoteattr`, which both quotes and escapes it — a
    base URL carrying an ampersand would otherwise produce a document that is
    not XML.
    """
    url = quoteattr(stream_url(public_base_url, path))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f"<Stream url={url} />"
        "</Connect></Response>"
    )
