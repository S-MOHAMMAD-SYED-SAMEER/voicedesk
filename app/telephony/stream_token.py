"""Proving a media stream belongs to a call this process actually answered.

The webhook is signed; the WebSocket upgrade that follows it is not, and the
carrier offers nothing to sign it with. Without something, anybody who learns
the stream address can attach to it — and the only thing standing between
them and a conversation is that `_start` refuses a `CallSid` it has no row
for, which is a guess away from being satisfied.

So the webhook mints a token, embeds it in the `<Stream>` URL it hands back,
and the socket checks it. Three properties matter and each is deliberate:

* **Bound to one call.** The `CallSid` is inside the signed payload, so a
  token minted for one call cannot attach to another.
* **Short-lived.** An expiry is inside the payload too. A carrier connects
  within seconds of the webhook; a token that outlived the call would be a
  password that never changed.
* **Nothing new to keep.** It is an HMAC under the Twilio auth token, which
  this process already has and already treats as a secret. There is no store,
  no cleanup, and nothing to get out of step across replicas.

The token is a credential. It is never logged — not on success, not on
refusal, not in an error — and it is compared with `hmac.compare_digest`.
"""

import base64
import hmac
import time
from hashlib import sha256

# Version prefix, so the format can change later without a token from the old
# one being read as a valid token of the new shape.
VERSION = "v1"
SEPARATOR = "."


class StreamTokenError(Exception):
    """A token could not be minted."""


def mint(auth_token: str, call_sid: str, ttl_seconds: int, *, now: float | None = None) -> str:
    """A token for one call, good until it expires."""
    if not auth_token:
        raise StreamTokenError(
            "No Twilio auth token is configured, so a stream URL cannot be "
            "signed."
        )
    if not call_sid:
        raise StreamTokenError("A stream token needs the call it belongs to.")

    expires = int((time.time() if now is None else now) + ttl_seconds)
    return f"{VERSION}{SEPARATOR}{expires}{SEPARATOR}{_sign(auth_token, call_sid, expires)}"


def is_valid(
    auth_token: str, call_sid: str, token: str | None, *, now: float | None = None
) -> bool:
    """Whether this token was minted here, for this call, and is still good.

    Fails closed on everything: no token, no auth token, a malformed one, an
    expired one, or one minted for a different call.
    """
    if not token or not auth_token or not call_sid:
        return False

    parts = token.split(SEPARATOR)
    if len(parts) != 3 or parts[0] != VERSION:
        return False

    _, raw_expiry, signature = parts
    try:
        expires = int(raw_expiry)
    except ValueError:
        return False

    if (time.time() if now is None else now) > expires:
        return False

    return hmac.compare_digest(_sign(auth_token, call_sid, expires), signature)


def _sign(auth_token: str, call_sid: str, expires: int) -> str:
    payload = f"{VERSION}{SEPARATOR}{call_sid}{SEPARATOR}{expires}".encode()
    digest = hmac.new(auth_token.encode("utf-8"), payload, sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


__all__ = ["SEPARATOR", "VERSION", "StreamTokenError", "is_valid", "mint"]
