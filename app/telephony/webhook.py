"""The carrier's first request: somebody is calling.

    carrier → POST /telephony/voice → signature check → Call row → TwiML

That is all this endpoint does. It answers with an instruction to open a media
stream, and everything that happens on the call happens on that socket.

The order matters. The signature is checked before anything is written, so an
unsigned request cannot create a row, cannot reserve a call identifier, and
cannot cost anything.

The body is parsed here rather than through FastAPI's form handling, which
needs a multipart library this project does not have and does not need: a
voice webhook is always `application/x-www-form-urlencoded`, which is
`urllib.parse`. It also keeps the signature check honest — the carrier signs
the parameters exactly as they arrived.
"""

import logging
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_sessionmaker
from app.models import Call, CallDirection
from app.telephony.security import SIGNATURE_HEADER, is_valid_signature
from app.telephony.twiml import TwiMLError, connect_stream

logger = logging.getLogger(__name__)

router = APIRouter(tags=["telephony"])

TWIML_MEDIA_TYPE = "application/xml"


@router.post("/telephony/voice", response_class=Response)
async def voice(request: Request) -> Response:
    """Answer an inbound call by opening a media stream back to us."""
    settings = get_settings()
    if not settings.telephony_enabled:
        # Not merely inert: absent. There is nothing here to probe.
        raise HTTPException(status_code=404, detail="Not Found")

    params = _parameters(await request.body())

    if settings.validate_twilio_signature:
        if not is_valid_signature(
            settings.twilio_auth_token,
            _signed_url(request, settings),
            params,
            request.headers.get(SIGNATURE_HEADER),
        ):
            # Nothing about the token, and nothing about why. An attacker
            # learning which half was wrong is an attacker making progress.
            logger.warning("Refused an unsigned request to the voice webhook.")
            raise HTTPException(status_code=403, detail="Invalid signature")

    call_sid = params.get("CallSid", "").strip()
    if not call_sid:
        raise HTTPException(status_code=400, detail="No CallSid")

    try:
        body = connect_stream(settings.public_base_url)
    except TwiMLError as exc:
        logger.error("Cannot answer a call: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    with get_sessionmaker()() as session:
        _record_call(session, call_sid, params, settings)

    return Response(content=body, media_type=TWIML_MEDIA_TYPE)


def _parameters(body: bytes) -> dict[str, str]:
    """The posted form, as the carrier sent it.

    `keep_blank_values` matters: the carrier includes empty fields and signs
    them, so dropping one would change the signature.
    """
    try:
        pairs = parse_qsl(body.decode("utf-8"), keep_blank_values=True)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Unreadable body") from exc
    return dict(pairs)


def _signed_url(request: Request, settings: Settings) -> str:
    """The address the carrier believes it called.

    Behind a proxy the request's own scheme and host describe the last hop,
    not the address on the internet the carrier dialled and signed. When a
    public base URL is configured it is the authority; otherwise the request
    has to speak for itself.
    """
    if not settings.public_base_url:
        return str(request.url)

    base = urlsplit(settings.public_base_url.rstrip("/"))
    return urlunsplit(
        (
            base.scheme or "https",
            base.netloc,
            base.path + request.url.path,
            request.url.query,
            "",
        )
    )


def _record_call(
    session: Session, call_sid: str, params: dict[str, str], settings: Settings
) -> Call:
    """The call, written once however many times the carrier asks.

    Carriers retry. The unique constraint on `provider_call_sid` is what makes
    that safe, and this reads the existing row back rather than treating a
    retry as an error — the caller is on the line either way.
    """
    existing = session.execute(
        select(Call).where(Call.provider_call_sid == call_sid)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    call = Call(
        direction=CallDirection.INBOUND,
        from_number=params.get("From") or settings.twilio_phone_number or "unknown",
        to_number=params.get("To") or settings.twilio_phone_number or "unknown",
        provider_call_sid=call_sid,
    )
    session.add(call)
    try:
        session.commit()
    except IntegrityError:
        # Two retries arriving at once. The constraint settled it; read the
        # winner back rather than failing a caller's call over a race.
        session.rollback()
        return session.execute(
            select(Call).where(Call.provider_call_sid == call_sid)
        ).scalar_one()
    return call
