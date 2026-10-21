"""The staff view: what the team has to deal with.

    GET /staff            a plain page, newest first
    GET /staff/requests   the same rows as JSON

Messages callers left, requests that somebody call a caller back (there is no live transfer), and the record that the
system could not answer. Every promise the receptionist makes to a caller — "the team will call you back" — has a row
here behind it.

Read-only, and deliberately small: no sign-in, no assignment, no "mark as handled". Outside production it is open, as
the harness is. In production it is closed unless `VOICEDESK_STAFF_VIEW_TOKEN` is set, and then it needs that token as
a bearer credential (`Authorization: Bearer <token>`). A name and a telephone number are personal data, so it is never
served unprotected in production, and it is never linked from anywhere.
"""

import hmac
from html import escape
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_session
from app.models import StaffRequest

router = APIRouter(tags=["staff"], include_in_schema=False)

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def require_staff_access(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """404 when the view is closed (it is not there to probe); 401 without the token when one is configured."""
    if not settings.staff_view_available:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    token = settings.staff_view_token
    if token:
        supplied = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            )


def _recent(session: Session, limit: int) -> list[StaffRequest]:
    return list(
        session.execute(
            select(StaffRequest).order_by(StaffRequest.created_at.desc()).limit(limit)
        )
        .scalars()
        .all()
    )


def _as_json(item: StaffRequest) -> dict[str, object]:
    return {
        "id": str(item.id),
        "kind": str(item.kind),
        "created_at": item.created_at.isoformat(),
        "caller_name": item.caller_name,
        "phone": item.phone,
        "body": item.body,
        "call_id": str(item.call_id) if item.call_id else None,
        "handled_at": item.handled_at.isoformat() if item.handled_at else None,
    }


_LABELS = {
    "message": "Message",
    "transfer": "Asked for a person",
    "failure": "System could not answer",
}

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Staff view</title>
<style>
  body {{ font: 15px/1.45 system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  p.note {{ color: #555; margin: 0 0 18px; max-width: 60em; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 70em; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }}
  th {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #555; }}
  .kind {{ white-space: nowrap; font-weight: 600; }}
  .failure .kind {{ color: #a33; }}
  .empty {{ color: #555; padding: 18px 0; }}
</style>
</head>
<body>
<h1>Staff view</h1>
<p class="note">What callers asked the team to deal with, newest first. There is no live transfer: every
"asked for a person" row is a request to call the caller back.</p>
{body}
</body>
</html>
"""


def render(items: list[StaffRequest]) -> str:
    if not items:
        return '<p class="empty">Nothing waiting.</p>'
    rows = "\n".join(
        f'<tr class="{escape(str(item.kind))}">'
        f'<td class="kind">{escape(_LABELS.get(str(item.kind), str(item.kind)))}</td>'
        f"<td>{escape(item.created_at.strftime('%Y-%m-%d %H:%M'))} UTC</td>"
        f"<td>{escape(item.caller_name or '—')}</td>"
        f"<td>{escape(item.phone or '—')}</td>"
        f"<td>{escape(item.body)}</td>"
        "</tr>"
        for item in items
    )
    return (
        "<table><thead><tr><th>What</th><th>When</th><th>Name</th><th>Number</th>"
        f"<th>Message or reason</th></tr></thead><tbody>\n{rows}\n</tbody></table>"
    )


@router.get(
    "/staff",
    response_class=HTMLResponse,
    dependencies=[Depends(require_staff_access)],
)
def staff_page(
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(body=render(_recent(session, limit))))


@router.get("/staff/requests", dependencies=[Depends(require_staff_access)])
def staff_requests(
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> list[dict[str, object]]:
    return [_as_json(item) for item in _recent(session, limit)]
