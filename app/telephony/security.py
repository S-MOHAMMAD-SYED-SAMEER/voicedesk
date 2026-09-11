"""Proving a webhook really came from the carrier.

Without this, anyone who learns the URL can start calls against this database
and spend this account's model budget. The check is the carrier's documented
one — HMAC-SHA1 over the full URL followed by every POST parameter in sorted
order, base64-encoded — and it needs nothing but the standard library.

The auth token is a secret. It is never logged, never echoed in an error, and
compared with `hmac.compare_digest` so a wrong signature takes the same time
as a right one.
"""

import base64
import hmac
from hashlib import sha1
from collections.abc import Mapping

SIGNATURE_HEADER = "X-Twilio-Signature"


def expected_signature(auth_token: str, url: str, params: Mapping[str, str]) -> str:
    """What the signature should be for this request."""
    payload = url + "".join(
        f"{key}{params[key]}" for key in sorted(params)
    )
    digest = hmac.new(
        auth_token.encode("utf-8"), payload.encode("utf-8"), sha1
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def is_valid_signature(
    auth_token: str,
    url: str,
    params: Mapping[str, str],
    signature: str | None,
) -> bool:
    """Whether this request carries a signature only the carrier could produce.

    A missing signature and a missing token are both failures. Treating an
    unconfigured token as "nothing to check" would turn a deployment mistake
    into an open endpoint.
    """
    if not signature or not auth_token:
        return False
    return hmac.compare_digest(
        expected_signature(auth_token, url, params), signature
    )
