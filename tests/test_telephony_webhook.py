"""Answering an inbound call: prove it is the carrier, then open a stream."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Call, CallDirection
from app.telephony.security import SIGNATURE_HEADER, expected_signature

from .conftest import TWILIO_AUTH_TOKEN, TWILIO_CALL_SID

VOICE = "/telephony/voice"
FORM = {
    "CallSid": TWILIO_CALL_SID,
    "AccountSid": "AC1",
    "From": "+447700900123",
    "To": "+441234567890",
}


@pytest.fixture
def webhook(migrated_engine: Engine, monkeypatch: pytest.MonkeyPatch):
    """A client whose settings can be tuned per test."""
    from app.main import create_app

    def build(**overrides):
        fields = {
            "telephony_enabled": True,
            "twilio_auth_token": TWILIO_AUTH_TOKEN,
            "public_base_url": "https://voicedesk.example.com",
            "validate_twilio_signature": True,
        }
        fields.update(overrides)
        settings = Settings(_env_file=None, **fields)
        monkeypatch.setattr(
            "app.telephony.webhook.get_settings", lambda: settings
        )
        get_settings.cache_clear()
        monkeypatch.setattr("app.config.get_settings", lambda: settings)
        return TestClient(create_app())

    return build


# The carrier signs the address it dialled — the configured public URL, not
# whatever host the request arrived at after a proxy.
PUBLIC_URL = "https://voicedesk.example.com/telephony/voice"


def _signed(client: TestClient, form: dict | None = None, url: str = PUBLIC_URL, **overrides):
    body = dict(FORM if form is None else form)
    signature = expected_signature(TWILIO_AUTH_TOKEN, url, body)
    headers = {SIGNATURE_HEADER: overrides.pop("signature", signature)}
    return client.post(VOICE, data=body, headers=headers, **overrides)


# --- answering a call -----------------------------------------------------


def test_a_signed_request_is_answered_with_a_media_stream(webhook) -> None:
    response = _signed(webhook())

    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]
    assert "<Connect>" in response.text
    assert "wss://voicedesk.example.com/telephony/stream" in response.text


def test_a_signed_request_records_the_call(
    session: Session, webhook
) -> None:
    _signed(webhook())

    call = session.execute(select(Call)).scalar_one()
    assert call.provider_call_sid == TWILIO_CALL_SID
    assert call.from_number == "+447700900123"
    assert call.to_number == "+441234567890"
    assert call.direction is CallDirection.INBOUND
    assert call.started_at is not None


def test_a_new_call_has_not_ended_and_has_no_outcome(
    session: Session, webhook
) -> None:
    """Those belong to the end of the call, and to a later milestone."""
    _signed(webhook())

    call = session.execute(select(Call)).scalar_one()
    assert call.ended_at is None
    assert call.outcome is None
    assert call.total_cost_usd is None


def test_a_retried_webhook_does_not_create_a_second_call(
    session: Session, webhook
) -> None:
    """Carriers retry. One call is one row."""
    client = webhook()

    _signed(client)
    second = _signed(client)

    assert second.status_code == 200
    assert len(session.execute(select(Call)).scalars().all()) == 1


def test_the_browser_sentinels_are_untouched_by_all_this(
    session: Session, webhook
) -> None:
    """Two adapters write this table; neither changes the other's rows."""
    from app.api.harness import BROWSER_FROM_NUMBER, BROWSER_TO_NUMBER

    session.add(
        Call(
            direction=CallDirection.INBOUND,
            from_number=BROWSER_FROM_NUMBER,
            to_number=BROWSER_TO_NUMBER,
        )
    )
    session.commit()

    _signed(webhook())

    calls = session.execute(select(Call).order_by(Call.started_at)).scalars().all()
    assert [call.provider_call_sid for call in calls] == [None, TWILIO_CALL_SID]
    assert calls[0].from_number == "browser"


# --- refusing a call ------------------------------------------------------


def test_an_unsigned_request_is_refused(session: Session, webhook) -> None:
    response = webhook().post(VOICE, data=FORM)

    assert response.status_code == 403
    assert session.execute(select(Call)).first() is None


def test_a_wrongly_signed_request_is_refused(session: Session, webhook) -> None:
    response = _signed(webhook(), signature="not-the-signature")

    assert response.status_code == 403
    assert session.execute(select(Call)).first() is None


def test_a_tampered_parameter_is_refused(session: Session, webhook) -> None:
    """The signature covers the parameters, so changing one invalidates it."""
    client = webhook()
    signature = expected_signature(TWILIO_AUTH_TOKEN, PUBLIC_URL, FORM)

    response = client.post(
        VOICE,
        data=FORM | {"From": "+440000000000"},
        headers={SIGNATURE_HEADER: signature},
    )

    assert response.status_code == 403
    assert session.execute(select(Call)).first() is None


def test_an_unconfigured_token_refuses_everything(session: Session, webhook) -> None:
    """A deployment mistake must not become an open endpoint."""
    response = _signed(webhook(twilio_auth_token=""))

    assert response.status_code == 403
    assert session.execute(select(Call)).first() is None


def test_a_request_with_no_call_identifier_is_refused(
    session: Session, webhook
) -> None:
    response = _signed(webhook(), form={"From": "+441", "To": "+442"})

    assert response.status_code == 400
    assert session.execute(select(Call)).first() is None


def test_no_public_url_means_the_call_cannot_be_answered(
    session: Session, webhook
) -> None:
    """Better to fail loudly than send a carrier an address it cannot reach.

    With no public URL configured the request has to speak for itself, so the
    signature here covers the address the request actually arrived at.
    """
    response = _signed(
        webhook(public_base_url=""), url="http://testserver/telephony/voice"
    )

    assert response.status_code == 503
    assert session.execute(select(Call)).first() is None


# --- switched off ---------------------------------------------------------


def test_the_endpoint_is_absent_until_telephony_is_enabled(
    session: Session, webhook
) -> None:
    """Not inert — absent. There is nothing here to probe."""
    response = _signed(webhook(telephony_enabled=False))

    assert response.status_code == 404
    assert session.execute(select(Call)).first() is None


def test_signature_checking_can_be_disabled_for_local_replay(
    session: Session, webhook
) -> None:
    response = webhook(validate_twilio_signature=False).post(VOICE, data=FORM)

    assert response.status_code == 200
    assert session.execute(select(Call)).scalar_one().provider_call_sid == TWILIO_CALL_SID


def test_the_signature_is_checked_against_the_public_address(
    session: Session, webhook
) -> None:
    """Signing the proxy's hostname instead of the public one is rejected."""
    response = _signed(webhook(), url="http://testserver/telephony/voice")

    assert response.status_code == 403
    assert session.execute(select(Call)).first() is None
