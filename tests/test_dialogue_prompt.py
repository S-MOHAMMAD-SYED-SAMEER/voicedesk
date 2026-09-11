"""The system prompt: the rules it states, and the menu it is given."""

import re

from sqlalchemy.orm import Session

from app.dialogue.prompt import (
    NO_SERVICES,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    build_system_prompt,
)
from app.models import Service


def test_the_version_is_a_stable_constant() -> None:
    assert SYSTEM_PROMPT_VERSION == "m4.1"


def test_the_prompt_never_invents_availability() -> None:
    assert "Never invent availability" in SYSTEM_PROMPT
    assert "check_availability" in SYSTEM_PROMPT


def test_the_prompt_requires_checking_before_offering_a_time() -> None:
    assert "Never book a time you have not checked" in SYSTEM_PROMPT


def test_the_prompt_requires_confirmation_before_booking() -> None:
    assert "get their agreement before you call `book_appointment`" in SYSTEM_PROMPT


def test_the_prompt_forbids_claiming_a_tool_succeeded_when_it_did_not() -> None:
    rule = "Never say a booking, reschedule or cancellation succeeded"
    assert rule in SYSTEM_PROMPT


def test_the_prompt_forbids_prices() -> None:
    """There is no price column, so there is no price it could be right about."""
    assert "Never state a price" in SYSTEM_PROMPT
    assert "no pricing information" in SYSTEM_PROMPT


def test_the_prompt_asks_rather_than_guesses() -> None:
    assert "Ask for anything you are missing" in SYSTEM_PROMPT


def test_the_prompt_keeps_replies_short() -> None:
    assert "two sentences" in SYSTEM_PROMPT


def test_the_prompt_covers_every_escalation_trigger() -> None:
    assert "transfer_to_human" in SYSTEM_PROMPT
    for trigger in (
        "asks to speak to a person",
        "medical or legal advice",
        "complaining, or wants a refund",
        "asked the same thing twice",
    ):
        assert trigger in SYSTEM_PROMPT, trigger


def test_the_prompt_offers_a_message_as_the_other_way_out() -> None:
    assert "take_message" in SYSTEM_PROMPT


def test_the_prompt_hides_the_machinery_from_the_caller() -> None:
    assert "Never mention tools, databases, errors, identifiers" in SYSTEM_PROMPT


def test_the_prompt_says_nothing_about_speech_confidence() -> None:
    """Low-confidence handling needs speech, which milestone 4 does not have."""
    lowered = SYSTEM_PROMPT.lower()
    for absent in ("stt", "speech", "transcription", "confidence", "garbled"):
        assert absent not in lowered, absent


def test_the_prompt_is_byte_stable(session: Session, haircut: Service) -> None:
    """No timestamps, no identifiers: the same call produces the same bytes."""
    assert build_system_prompt(session) == build_system_prompt(session)


def test_the_prompt_contains_nothing_that_varies_per_call(
    session: Session, haircut: Service
) -> None:
    prompt = build_system_prompt(session)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:", prompt)
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-", prompt)


def test_the_menu_lists_the_active_services(
    session: Session, haircut: Service
) -> None:
    session.add(Service(name="Beard trim", duration_minutes=15, staff_id="sam"))
    session.commit()

    prompt = build_system_prompt(session)

    assert "- Haircut" in prompt
    assert "- Beard trim" in prompt


def test_the_menu_excludes_inactive_services(
    session: Session, haircut: Service
) -> None:
    """An inactive service is not bookable, so naming it invites a refusal."""
    session.add(
        Service(name="Perm", duration_minutes=90, staff_id="sam", active=False)
    )
    session.commit()

    assert "Perm" not in build_system_prompt(session)


def test_the_menu_carries_no_durations_or_hours(
    session: Session, haircut: Service
) -> None:
    """Those stay in the database and the calendar, with one answer each."""
    menu = build_system_prompt(session).split("use these names exactly:")[1]

    assert "30" not in menu
    assert "minute" not in menu.lower()
    assert "09:00" not in menu


def test_no_services_configured_says_so_rather_than_leaving_a_blank(
    session: Session,
) -> None:
    assert NO_SERVICES in build_system_prompt(session)
