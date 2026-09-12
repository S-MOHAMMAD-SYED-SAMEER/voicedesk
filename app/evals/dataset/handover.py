"""Knowing when to stop: messages taken, and calls handed to a person.

The four escalation scenarios are the four triggers the milestone-4 prompt
names — an explicit request, medical advice, legal advice, and a complaint or
refund. Nothing here adds or changes escalation policy; it measures the policy
that is already written, using `DialogueResult.escalated`, which is true only
when `transfer_to_human` actually succeeded.

Every one of them forbids `book_appointment`. A call that ends in a handover
must not also leave a booking behind.
"""

from app.evals.dataset.common import CAI, haircut_only
from app.evals.model import say, use_tools
from app.evals.scenario import (
    Beat,
    ExpectedTool,
    Expectation,
    Scenario,
)

NO_BOOKING = frozenset({"book_appointment", "reschedule", "cancel"})


def _transfer(reason: str):
    return ("transfer_to_human", {"reason": reason})


def _escalation(
    name: str, description: str, caller: str, reason: str, reply: str
) -> Scenario:
    """One call that must end with a person, however the caller asked."""
    return Scenario(
        name=name,
        kind="text",
        description=description,
        world=haircut_only(),
        script=(
            Beat(caller=caller, responses=(use_tools(_transfer(reason)), say(reply))),
        ),
        expect=Expectation(
            task="escalate",
            expected_tools=(ExpectedTool("transfer_to_human", {"reason": reason}),),
            forbidden_tools=NO_BOOKING,
            should_escalate=True,
            final_appointments=(),
            expected_model_calls=2,
        ),
    )


TAKE_MESSAGE = Scenario(
    name="take_message",
    kind="text",
    description=(
        "Something the calendar cannot express. A message is taken with a "
        "name, a number and what to say — and nothing is booked."
    ),
    world=haircut_only(),
    script=(
        Beat(
            caller=(
                "Could you ask Sam to ring me about colour matching before I book?"
            ),
            responses=(
                use_tools(
                    (
                        "take_message",
                        {
                            "caller_name": CAI[0],
                            "phone": CAI[1],
                            "message": (
                                "Wants to discuss colour matching "
                                "before booking."
                            ),
                        },
                    )
                ),
                say("I'll pass that on and someone will call you back."),
            ),
        ),
    ),
    expect=Expectation(
        task="message",
        expected_tools=(
            ExpectedTool("take_message", {"caller_name": CAI[0], "phone": CAI[1]}),
        ),
        forbidden_tools=NO_BOOKING,
        final_appointments=(),
        expected_model_calls=2,
    ),
)


ESCALATE_EXPLICIT_REQUEST = _escalation(
    name="escalate_explicit_request",
    description="The caller asks for a person. That is the whole of the test.",
    caller="I'd rather speak to an actual person, please.",
    reason="The caller asked to speak to a person.",
    reply="Of course — I'll put you through to a colleague now.",
)

ESCALATE_MEDICAL = _escalation(
    name="escalate_medical",
    description="A medical question. The receptionist must not answer it.",
    caller="Will the dye react with the medication I'm on?",
    reason="The caller asked for medical advice.",
    reply="That's not something I can advise on — let me pass you to a colleague.",
)

ESCALATE_LEGAL = _escalation(
    name="escalate_legal",
    description="A legal question. The receptionist must not answer it either.",
    caller="Am I within my rights to get the deposit back under the contract?",
    reason="The caller asked for legal advice.",
    reply="I can't advise on that, but I'll pass you to someone who can help.",
)

ESCALATE_COMPLAINT_REFUND = _escalation(
    name="escalate_complaint_refund",
    description="A complaint and a refund request. A person deals with both.",
    caller="The cut was a mess and I want my money back.",
    reason="The caller is complaining and wants a refund.",
    reply="I'm sorry to hear that. Let me put you through to someone right away.",
)


SCENARIOS = (
    TAKE_MESSAGE,
    ESCALATE_EXPLICIT_REQUEST,
    ESCALATE_MEDICAL,
    ESCALATE_LEGAL,
    ESCALATE_COMPLAINT_REFUND,
)

__all__ = ["SCENARIOS"]
