"""Messages and callback requests are stored and shown to staff; the caller is never promised a live colleague."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.clock import fixed_clock
from app.config import Settings
from app.dialogue import LOOP_EXHAUSTED_REPLY, MODEL_FAILURE_REPLY, Conversation
from app.dialogue.prompt import SYSTEM_PROMPT
from app.main import create_app
from app.models import StaffRequest, StaffRequestKind
from app.providers.llm import ModelError, ModelUnavailable
from app.tools import ToolContext, take_message, transfer_to_human
from tests.conftest import FakeModel, say, use_tools

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def _rows(session) -> list[StaffRequest]:
    session.expire_all()
    return list(session.execute(select(StaffRequest).order_by(StaffRequest.created_at)).scalars())


# --- take_message -----------------------------------------------------------


def test_a_message_is_stored_with_everything_the_caller_gave(session, call, calendar_settings) -> None:
    context = ToolContext(session=session, settings=calendar_settings, call_id=call.id)

    result = take_message(
        context, caller_name="Ada Lovelace", phone="+447700900123", message="Can someone ring me about a quote?"
    )

    assert result.success and result.data["recorded"] is True
    (row,) = _rows(session)
    assert row.kind is StaffRequestKind.MESSAGE
    assert (row.caller_name, row.phone, row.body) == (
        "Ada Lovelace",
        "+447700900123",
        "Can someone ring me about a quote?",
    )
    assert row.call_id == call.id and row.handled_at is None


def test_a_message_missing_a_field_is_refused_and_stores_nothing(session, call, calendar_settings) -> None:
    context = ToolContext(session=session, settings=calendar_settings, call_id=call.id)

    result = take_message(context, caller_name="Ada", phone="", message="Hello")

    assert not result.success
    assert _rows(session) == []


def test_the_message_result_tells_the_model_what_it_may_say(session, call, calendar_settings) -> None:
    context = ToolContext(session=session, settings=calendar_settings, call_id=call.id)

    result = take_message(context, caller_name="Ada", phone="07700900123", message="Hello")

    assert "will call back" in result.data["what_to_tell_the_caller"]
    assert "Do not say anyone is on the line" in result.data["what_to_tell_the_caller"]


# --- transfer_to_human -------------------------------------------------------


def test_a_transfer_is_stored_as_a_callback_request_with_the_carriers_number(
    session, call, calendar_settings
) -> None:
    context = ToolContext(session=session, settings=calendar_settings, call_id=call.id)

    result = transfer_to_human(context, reason="The caller asked for medical advice.")

    assert result.success and result.data["escalated"] is True
    assert result.data["live_transfer_available"] is False
    (row,) = _rows(session)
    assert row.kind is StaffRequestKind.TRANSFER
    assert row.phone == call.from_number and row.caller_name is None
    assert row.body == "The caller asked for medical advice."


def test_the_transfer_result_forbids_saying_the_caller_is_being_put_through(
    session, call, calendar_settings
) -> None:
    context = ToolContext(session=session, settings=calendar_settings, call_id=call.id)

    text = transfer_to_human(context, reason="Wants a person.").data["what_to_tell_the_caller"]

    assert "Never say you are transferring" in text and "call back" in text


def test_a_transfer_without_a_reason_is_refused(session, call, calendar_settings) -> None:
    context = ToolContext(session=session, settings=calendar_settings, call_id=call.id)

    assert not transfer_to_human(context, reason="  ").success
    assert _rows(session) == []


# --- what the caller is told -------------------------------------------------


@pytest.mark.parametrize("reply", [MODEL_FAILURE_REPLY, LOOP_EXHAUSTED_REPLY])
def test_a_failure_reply_never_promises_a_person_on_the_line(reply: str) -> None:
    lowered = reply.lower()

    assert "colleague" not in lowered and "put you through" not in lowered
    assert "call you back" in lowered


def test_the_prompt_says_nobody_can_take_the_call_and_no_one_is_being_put_through() -> None:
    lowered = SYSTEM_PROMPT.lower()

    assert "cannot connect a caller to a person" in lowered
    assert "never say you are transferring" in lowered
    assert "call back" in lowered
    assert "pass the caller to a colleague" not in lowered


def test_a_model_failure_is_backed_by_a_request_that_the_team_call_back(
    session, call, calendar_settings
) -> None:
    model = FakeModel(raises=ModelUnavailable("down"))

    result = Conversation(session, call, model, calendar_settings, clock=fixed_clock(NOW)).send("Hello?")

    assert result.failed and result.text == MODEL_FAILURE_REPLY
    (row,) = _rows(session)
    assert row.kind is StaffRequestKind.FAILURE
    assert row.phone == call.from_number and row.call_id == call.id
    assert "could not answer" in row.body


def test_an_exhausted_tool_loop_is_backed_by_a_request_too(session, call, open_weekdays, haircut) -> None:
    settings = Settings(_env_file=None, business_timezone="UTC", max_tool_iterations=2)
    looping = [use_tools(("check_availability", {"service_name": "Haircut", "day": "2026-03-02"}))] * 2

    result = Conversation(session, call, FakeModel(*looping), settings, clock=fixed_clock(NOW)).send("?")

    assert result.failed and result.text == LOOP_EXHAUSTED_REPLY
    (row,) = _rows(session)
    assert row.kind is StaffRequestKind.FAILURE and "tool loop" in row.body


def test_a_successful_turn_stores_no_staff_request(session, call, calendar_settings) -> None:
    Conversation(session, call, FakeModel(say("Hello.")), calendar_settings, clock=fixed_clock(NOW)).send("Hi")

    assert _rows(session) == []


def test_a_refusal_is_a_failure_with_a_request(session, call, calendar_settings) -> None:
    from app.providers.llm import ModelRefused

    result = Conversation(
        session, call, FakeModel(raises=ModelRefused("no")), calendar_settings, clock=fixed_clock(NOW)
    ).send("Hi")

    assert result.failed and len(_rows(session)) == 1
    assert issubclass(ModelRefused, ModelError)


# --- the staff view ----------------------------------------------------------


@pytest.fixture
def staff_client(migrated_engine, monkeypatch):
    """The app against the test database, outside production."""
    from app.config import get_settings
    from app.db.session import reset_engine

    def build(**environment: str) -> TestClient:
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        get_settings.cache_clear()
        reset_engine()
        return TestClient(create_app())

    return build


def _store(session, **fields) -> None:
    session.add(StaffRequest(**fields))
    session.commit()


def test_the_staff_view_lists_requests_newest_first(session, staff_client) -> None:
    _store(session, kind=StaffRequestKind.MESSAGE, caller_name="Ada", phone="07700900123", body="First")
    _store(session, kind=StaffRequestKind.TRANSFER, phone="+447700900456", body="Second")

    response = staff_client().get("/staff")

    assert response.status_code == 200
    page = response.text
    assert page.index("Second") < page.index("First")
    assert "Asked for a person" in page and "Message" in page and "Ada" in page


def test_the_staff_view_escapes_what_a_caller_said(session, staff_client) -> None:
    _store(session, kind=StaffRequestKind.MESSAGE, caller_name="<b>Eve</b>", phone="1", body="<script>alert(1)</script>")

    page = staff_client().get("/staff").text

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page and "&lt;b&gt;Eve" in page


def test_the_staff_view_says_so_when_nothing_is_waiting(session, staff_client) -> None:
    assert "Nothing waiting" in staff_client().get("/staff").text


def test_the_json_view_carries_the_same_rows(session, staff_client) -> None:
    _store(session, kind=StaffRequestKind.FAILURE, phone="+447700900123", body="The system could not answer: x.")

    rows = staff_client().get("/staff/requests").json()

    assert rows[0]["kind"] == "failure" and rows[0]["phone"] == "+447700900123"
    assert rows[0]["handled_at"] is None


def test_in_production_the_staff_view_is_absent_without_a_token(session, staff_client) -> None:
    client = staff_client(
        VOICEDESK_ENVIRONMENT="production",
        VOICEDESK_ANTHROPIC_API_KEY="sk-test",
    )

    assert client.get("/staff").status_code == 404
    assert client.get("/staff/requests").status_code == 404


def test_a_configured_token_is_required(session, staff_client) -> None:
    client = staff_client(VOICEDESK_STAFF_VIEW_TOKEN="s3cret-token")

    assert client.get("/staff").status_code == 401
    assert client.get("/staff", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/staff", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200
    assert client.get("/staff/requests", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200


def test_the_staff_view_can_be_switched_off(session, staff_client) -> None:
    assert staff_client(VOICEDESK_STAFF_VIEW_ENABLED="false").get("/staff").status_code == 404


def test_the_staff_view_availability_rules() -> None:
    assert Settings(_env_file=None).staff_view_available
    assert not Settings(_env_file=None, environment="production").staff_view_available
    assert Settings(
        _env_file=None, environment="production", staff_view_token="t"
    ).staff_view_available
    assert not Settings(_env_file=None, staff_view_enabled=False).staff_view_available
