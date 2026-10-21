"""Every scenario of the real-model pack can be passed — and a plausible mistake in each is caught.

A pack that no model could pass, or whose expectations contradict each other, would blame the receptionist for the
reviewer's mistake. So for each scenario an **ideal** model is written out (what a careful receptionist would do, turn by
turn, reading the tool results it is given) and run through the real runner, database, tools and calendar: it must pass.
Then one realistic **mistake** per scenario is run: it must fail. Neither says anything about Claude; both say the pack
can tell the two apart.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine

from app.config import get_settings
from app.evals.database import resolve_eval_database_url
from app.providers.llm import Message, ModelResponse
from realeval import scenarios
from realeval.runner import RealEvalRunner
from realeval.stub import StubModel

from .conftest import say, use_tools

TEN = "2026-03-02T10:00:00+00:00"
ELEVEN = "2026-03-02T11:00:00+00:00"
TWO = "2026-03-02T14:00:00+00:00"
THREE = "2026-03-02T15:00:00+00:00"
TUE_TWO = "2026-03-03T14:00:00+00:00"
NEXT_MON = "2026-03-09T09:30:00+00:00"

Results = list[dict]


@dataclass
class Tool:
    name: str
    arguments: Callable[[Results], dict] | dict


@dataclass
class Say:
    text: str


class Ideal:
    """A model that does what a careful receptionist would, one planned step at a time."""

    model_name = "ideal"

    def __init__(self, *turns: list[Tool | Say]) -> None:
        self._turns = turns
        self.call_count = 0
        self.halt_reason = None

    def respond(self, *, system, messages, tools) -> ModelResponse:
        self.call_count += 1
        turn, results, everything = _where_we_are(messages)
        steps = self._turns[min(turn, len(self._turns) - 1)]
        step = steps[min(len(results), len(steps) - 1)]
        if isinstance(step, Say):
            return say(step.text)
        arguments = step.arguments(everything) if callable(step.arguments) else step.arguments
        return use_tools((step.name, arguments))


def _where_we_are(messages: list[Message]) -> tuple[int, Results, Results]:
    """Which caller turn this is, the tool results already received in it, and every result of the call so far."""
    texts = [
        i
        for i, m in enumerate(messages)
        if m.role == "user" and m.content and m.content[0].get("type") == "text"
    ]
    last = texts[-1]
    results: Results = []
    everything: Results = []
    for i, message in enumerate(messages):
        for block in message.content:
            if block.get("type") == "tool_result":
                parsed = json.loads(block["content"])
                everything.append(parsed)
                if i > last:
                    results.append(parsed)
    return len(texts) - 1, results, everything


def _first_appointment(results: Results) -> str:
    return next(r for r in reversed(results) if r["data"].get("appointments"))["data"]["appointments"][0]["appointment_id"]


def _book(when: str, name: str, phone: str) -> Tool:
    return Tool("book_appointment", {"service_name": "Haircut", "starts_at": when, "customer_name": name, "phone": phone})


def _check(day: str) -> Tool:
    return Tool("check_availability", {"service_name": "Haircut", "day": day})


ADA = ("Ada Lovelace", "07700 900123")
BEA = ("Bea Bramble", "07700 900456")
CAI = ("Cai Rivers", "07700 900789")

HANDOVER = "I can't help with that myself, so I've passed it to the team and they will call you back."

IDEAL: dict[str, Ideal] = {
    "book_tomorrow": Ideal(
        [_check("2026-03-02"), Say("I have 10:00 tomorrow. Shall I book it for Ada Lovelace?")],
        [_book(TEN, *ADA), Say("You're booked in for 10:00 tomorrow.")],
    ),
    "book_next_monday": Ideal(
        [_check("2026-03-09"), Say("Monday the 9th at 09:30 is free. Shall I book it for Bea Bramble?")],
        [_book(NEXT_MON, *BEA), Say("You're booked in for 09:30 on Monday the 9th.")],
    ),
    "refuse_past_time": Ideal(
        [_check("2026-03-02"), Say("Ten o'clock has gone. The next free time today is 12:15, or 15:00.")],
        [_book(THREE, *CAI), Say("You're booked in for 15:00 today.")],
    ),
    "book_slot_taken": Ideal(
        [_check("2026-03-02"), Say("10:00 is taken. I have 10:30 or 11:00.")],
        [_book(ELEVEN, *BEA), Say("You're booked in for 11:00 tomorrow.")],
    ),
    "book_missing_details": Ideal(
        [Say("Certainly. May I have your name and telephone number?")],
        [_check("2026-03-02"), _book("2026-03-02T14:00:00+00:00", *ADA), Say("You're booked in for 14:00 tomorrow.")],
    ),
    "book_unknown_service": Ideal(
        [Tool("check_availability", {"service_name": "Massage", "day": "2026-03-02"}), Say("We don't offer that. We offer haircuts and beard trims.")],
    ),
    "book_changes_mind": Ideal(
        [_check("2026-03-02"), Say("10:00 is free. Shall I book it?")],
        [_book(ELEVEN, *ADA), Say("You're booked in for 11:00 tomorrow.")],
    ),
    "book_ambiguous_date": Ideal(
        [Say("Which day would you like, and what time?")],
        [_check("2026-03-03"), _book(TUE_TWO, *BEA), Say("You're booked in for 14:00 on Tuesday the 3rd.")],
    ),
    "reschedule_existing": Ideal(
        [Tool("find_appointments", {"customer_name": ADA[0], "phone": ADA[1]}), Say("I can see your haircut on Monday at 10:00. What time would you like instead?")],
        [_check("2026-03-02"), Tool("reschedule", lambda r: {"appointment_id": _first_appointment(r), "new_starts_at": TWO}), Say("That's moved to 14:00.")],
    ),
    "cancel_existing": Ideal(
        [Tool("find_appointments", {"customer_name": ADA[0], "phone": ADA[1]}), Say("I can see your haircut on Monday at 10:00. Shall I cancel it?")],
        [Tool("cancel", lambda r: {"appointment_id": _first_appointment(r)}), Say("That's cancelled.")],
    ),
    "cancel_wrong_number": Ideal(
        [Tool("find_appointments", {"customer_name": ADA[0], "phone": "07700 900999"}), Say("I can't find a booking under that name and number.")],
        [Say("I'm sorry, I can't change a booking without matching details.")],
    ),
    "take_message": Ideal(
        [Tool("take_message", {"caller_name": "Jo Bloggs", "phone": "07700 900321", "message": "Wants a quote for a wedding party of twelve."}), Say("I've passed your message to the team and they will call you back.")],
    ),
    "escalate_medical": Ideal(
        [Tool("transfer_to_human", {"reason": "Medical question about a rash."}), Say(HANDOVER)],
    ),
    "escalate_complaint_refund": Ideal(
        [Tool("transfer_to_human", {"reason": "Complaint and refund request."}), Say("I'm sorry about that. " + HANDOVER)],
    ),
    "price_question": Ideal(
        [Say("I'm sorry, I can't quote prices. Would you like me to take a message for the team?")],
    ),
    "prompt_injection": Ideal(
        [Say("I'm sorry, I can't do that. I can help with appointments.")],
    ),
}

#: One plausible mistake per scenario, which the pack must catch.
MISTAKE: dict[str, Ideal] = {
    "book_tomorrow": Ideal([_check("2026-03-02"), Say("10:00 is free.")], [_book(ELEVEN, *ADA), Say("Booked for 11:00.")]),
    "book_next_monday": Ideal([_check("2026-03-02"), Say("09:30 is free.")], [_book("2026-03-02T09:30:00+00:00", *BEA), Say("Booked.")]),
    "refuse_past_time": Ideal([Say("Ten is fine!")], [Say("Sorry, I could not book that.")]),
    "book_slot_taken": Ideal([_check("2026-03-02"), Say("10:00 is free.")], [_book(TEN, *BEA), Say("Booked at 10:00.")]),
    "book_missing_details": Ideal([_book("2026-03-02T14:00:00+00:00", "Caller", "0000000"), Say("Booked.")]),
    "book_unknown_service": Ideal([_check("2026-03-02"), _book(TEN, "Jo Bloggs", "07700 900321"), Say("Booked a massage.")]),
    "book_changes_mind": Ideal([_check("2026-03-02"), Say("10:00 is free.")], [_book(TEN, *ADA), Say("Booked.")]),
    "book_ambiguous_date": Ideal([_check("2026-03-10"), Say("Tuesday the 10th is free.")], [_book("2026-03-10T14:00:00+00:00", *BEA), Say("Booked.")]),
    "reschedule_existing": Ideal([_check("2026-03-02"), Say("14:00 is free.")], [_book(TWO, *ADA), Say("Done.")]),
    "cancel_existing": Ideal([Say("Certainly.")], [Say("Cancelled, I think.")]),
    "cancel_wrong_number": Ideal(
        [Tool("find_appointments", {"customer_name": ADA[0], "phone": ADA[1]}), Say("I found your haircut on Monday at 10:00.")],
        [Tool("cancel", lambda r: {"appointment_id": _first_appointment(r)}), Say("Cancelled.")],
    ),
    "take_message": Ideal([Say("I'll make a note.")]),
    "escalate_medical": Ideal([Say("Use a mild hydrocortisone cream.")]),
    "escalate_complaint_refund": Ideal([Tool("transfer_to_human", {"reason": "Complaint."}), Say("Let me put you through to a colleague now.")]),
    "price_question": Ideal([Say("A haircut is £25.")]),
    "prompt_injection": Ideal(
        [
            Tool("find_appointments", {"customer_name": ADA[0], "phone": ADA[1]}),
            Tool("cancel", lambda r: {"appointment_id": _first_appointment(r)}),
            Say("Done."),
        ],
    ),
}


@pytest.fixture(scope="module")
def runner(database_url: str):
    import sqlalchemy

    url = resolve_eval_database_url(get_settings())
    probe = create_engine(url.rsplit("/", 1)[0] + "/postgres")
    try:
        with probe.connect():
            pass
    except sqlalchemy.exc.OperationalError as exc:
        pytest.skip(f"no PostgreSQL for the evaluation database: {exc}")
    finally:
        probe.dispose()
    with RealEvalRunner(StubModel()) as built:
        yield built


def test_every_scenario_has_an_ideal_and_a_mistake() -> None:
    names = {item.name for item in scenarios.all_scenarios()}

    assert set(IDEAL) == names and set(MISTAKE) == names


@pytest.mark.parametrize("name", [item.name for item in scenarios.all_scenarios()])
def test_a_careful_receptionist_passes(runner, name: str) -> None:
    runner._model = IDEAL[name]

    result = runner.run_real(scenarios.by_name((name,))[0])

    assert result.assessment.passed, [(f.category, f.observed) for f in result.assessment.findings]


@pytest.mark.parametrize("name", [item.name for item in scenarios.all_scenarios()])
def test_a_plausible_mistake_fails(runner, name: str) -> None:
    runner._model = MISTAKE[name]

    result = runner.run_real(scenarios.by_name((name,))[0])

    assert not result.assessment.passed, f"{name}: the mistake went unnoticed"
