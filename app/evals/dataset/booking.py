"""Booking, and the seven ways it can go.

Four of these are adversarial: the script deliberately asks for something the
system must refuse — a slot that was never offered, a slot somebody else has
just taken, a service that does not exist, a booking with no name on it — and
the scenario passes only if the refusal happened and the database is unchanged.
A suite made only of happy paths would measure nothing.
"""

from app.evals.dataset.common import (
    ADA,
    BEA,
    HAIRCUT,
    MONDAY,
    NINE,
    NINE_THIRTY,
    TEN,
    ELEVEN,
    haircut_only,
    salon,
)
from app.evals.model import say, use_tools
from app.evals.scenario import (
    AppointmentSpec,
    Beat,
    Claim,
    ExpectedTool,
    Expectation,
    Scenario,
    SeedAppointment,
    ServiceSpec,
    World,
    weekdays,
)


def _check(service: str = HAIRCUT, day: str = MONDAY):
    return ("check_availability", {"service_name": service, "day": day})


def _book(starts_at: str, who: tuple[str, str] = ADA, service: str = HAIRCUT):
    name, phone = who
    return (
        "book_appointment",
        {
            "service_name": service,
            "starts_at": starts_at,
            "customer_name": name,
            "phone": phone,
        },
    )


BOOKING_AVAILABLE = Scenario(
    name="booking_available",
    kind="text",
    description="A caller asks for a haircut, is offered real times, and takes one.",
    world=haircut_only(),
    script=(
        Beat(
            caller="I'd like a haircut on Monday, please.",
            responses=(
                use_tools(_check()),
                say("I have 09:00, 09:30 or 10:00 free. Which suits you?"),
            ),
        ),
        Beat(
            caller="10:00 please. It's Ada Lovelace, 07700 900123.",
            responses=(
                use_tools(_book(TEN)),
                say("That's booked for 10:00 on Monday, Ada."),
            ),
        ),
    ),
    expect=Expectation(
        task="book",
        expected_tools=(
            ExpectedTool(
                "check_availability",
                {"service_name": HAIRCUT, "day": MONDAY},
            ),
            ExpectedTool(
                "book_appointment",
                {"service_name": HAIRCUT, "starts_at": TEN, "customer_name": ADA[0]},
            ),
        ),
        availability_claims=(
            Claim(HAIRCUT, NINE, 0),
            Claim(HAIRCUT, NINE_THIRTY, 0),
            Claim(HAIRCUT, TEN, 0),
        ),
        final_appointments=(AppointmentSpec(HAIRCUT, ADA[0], TEN),),
        expected_model_calls=4,
    ),
)


BOOKING_CALLER_CHANGES_MIND = Scenario(
    name="booking_caller_changes_mind",
    kind="text",
    description="The caller is offered two times, asks for one, then switches.",
    world=haircut_only(),
    script=(
        Beat(
            caller="Monday for a haircut, whatever you have.",
            responses=(
                use_tools(_check()),
                say("I have 10:00 and 11:00 free on Monday."),
            ),
        ),
        Beat(
            caller="10:00 — no, actually, make it 11:00. Ada Lovelace.",
            responses=(
                use_tools(_book(ELEVEN)),
                say("Booked for 11:00 on Monday, Ada."),
            ),
        ),
    ),
    expect=Expectation(
        task="book",
        expected_tools=(
            ExpectedTool("check_availability", {"service_name": HAIRCUT}),
            ExpectedTool("book_appointment", {"starts_at": ELEVEN}),
        ),
        availability_claims=(Claim(HAIRCUT, TEN, 0), Claim(HAIRCUT, ELEVEN, 0)),
        # Exactly one. A change of mind must not leave the first time held.
        final_appointments=(AppointmentSpec(HAIRCUT, ADA[0], ELEVEN),),
        expected_model_calls=4,
    ),
)


BOOKING_UNAVAILABLE_SLOT = Scenario(
    name="booking_unavailable_slot",
    kind="text",
    description=(
        "The caller insists on a time somebody else already has. The script "
        "tries to book it anyway and the executor's guard must refuse."
    ),
    world=World(
        services=(ServiceSpec(HAIRCUT, 30, "sam"),),
        hours=weekdays(),
        appointments=(
            SeedAppointment("taken", HAIRCUT, BEA[0], BEA[1], TEN),
        ),
    ),
    script=(
        Beat(
            caller="Can I have a haircut at ten on Monday?",
            responses=(
                use_tools(_check()),
                # No numeral for the time that is gone: the detector cannot
                # tell "free" from "taken", so the dataset never asks it to.
                say("That time has gone, I'm afraid. I have 09:00 or 11:00."),
            ),
        ),
        Beat(
            caller="No, I want ten.",
            responses=(
                use_tools(_book(TEN)),
                say("I can't do that time, sorry. Shall I put you down for 11:00?"),
            ),
        ),
    ),
    expect=Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool("check_availability", {"service_name": HAIRCUT}),
            ExpectedTool("book_appointment", {"starts_at": TEN}, succeeds=False),
        ),
        allow_tool_failures=frozenset({"book_appointment"}),
        availability_claims=(Claim(HAIRCUT, NINE, 0), Claim(HAIRCUT, ELEVEN, 0)),
        # Unchanged: the caller who already had 10:00 still has it.
        final_appointments=(AppointmentSpec(HAIRCUT, BEA[0], TEN),),
        expected_model_calls=4,
    ),
)


BOOKING_SLOT_TAKEN_BETWEEN_CHECK_AND_BOOK = Scenario(
    name="booking_slot_taken_between_check_and_book",
    kind="text",
    description=(
        "One slot, two callers, verified for both. The database's exclusion "
        "constraint decides, and the second booking is told plainly."
    ),
    world=haircut_only(),
    script=(
        Beat(
            caller="Monday please, for two of us.",
            responses=(
                use_tools(_check()),
                say("I have 09:00 and 10:00 free."),
            ),
        ),
        Beat(
            caller="10:00 for Ada Lovelace.",
            responses=(use_tools(_book(TEN, ADA)), say("Booked for 10:00, Ada.")),
        ),
        Beat(
            caller="And 10:00 for Bea Bramble as well.",
            responses=(
                use_tools(_book(TEN, BEA)),
                say("I can't put you both in then. I still have 09:00."),
            ),
        ),
    ),
    expect=Expectation(
        task="book",
        expected_tools=(
            ExpectedTool("check_availability", {"service_name": HAIRCUT}),
            ExpectedTool(
                "book_appointment", {"customer_name": ADA[0], "starts_at": TEN}
            ),
            ExpectedTool(
                "book_appointment",
                {"customer_name": BEA[0], "starts_at": TEN},
                succeeds=False,
            ),
        ),
        allow_tool_failures=frozenset({"book_appointment"}),
        availability_claims=(Claim(HAIRCUT, NINE, 0), Claim(HAIRCUT, TEN, 0)),
        final_appointments=(AppointmentSpec(HAIRCUT, ADA[0], TEN),),
        expected_model_calls=6,
    ),
)


BOOKING_UNKNOWN_SERVICE = Scenario(
    name="booking_unknown_service",
    kind="text",
    description=(
        "The caller asks for something the business does not do. Nothing may "
        "be invented, and nothing may be booked."
    ),
    world=haircut_only(),
    script=(
        Beat(
            caller="Do you do massages? I'd like one Monday.",
            responses=(
                use_tools(
                    (
                        "check_availability",
                        {"service_name": "Massage", "day": MONDAY},
                    )
                ),
                say("We don't offer that, I'm afraid. We do haircuts."),
            ),
        ),
        Beat(
            caller="A haircut then, what have you got?",
            responses=(
                use_tools(_check()),
                say("Monday has 09:00 and 10:00 free."),
            ),
        ),
    ),
    expect=Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool(
                "check_availability", {"service_name": "Massage"}, succeeds=False
            ),
            ExpectedTool("check_availability", {"service_name": HAIRCUT}),
        ),
        forbidden_tools=frozenset({"book_appointment"}),
        allow_tool_failures=frozenset({"check_availability"}),
        availability_claims=(Claim(HAIRCUT, NINE, 1), Claim(HAIRCUT, TEN, 1)),
        final_appointments=(),
        expected_model_calls=4,
    ),
)


BOOKING_AMBIGUOUS_SERVICE = Scenario(
    name="booking_ambiguous_service",
    kind="text",
    description=(
        "Two stylists offer a service of the same name. The lookup refuses to "
        "pick one, and nothing is booked on a guess."
    ),
    world=salon(
        ServiceSpec(HAIRCUT, 30, "sam"),
        ServiceSpec(HAIRCUT, 30, "ravi"),
    ),
    script=(
        Beat(
            caller="A haircut on Monday, please.",
            responses=(
                use_tools(_check()),
                say("We have two stylists doing that — did you want Sam or Ravi?"),
            ),
        ),
        Beat(
            caller="Oh, I'll check with my partner and call back.",
            responses=(say("Of course. We're here until five."),),
        ),
    ),
    expect=Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool(
                "check_availability", {"service_name": HAIRCUT}, succeeds=False
            ),
        ),
        forbidden_tools=frozenset({"book_appointment"}),
        allow_tool_failures=frozenset({"check_availability"}),
        final_appointments=(),
        expected_model_calls=3,
    ),
)


BOOKING_MISSING_DETAILS = Scenario(
    name="booking_missing_details",
    kind="text",
    description=(
        "The script tries to book with no name on it. The tool must refuse, "
        "and the booking must only happen once the caller has given one."
    ),
    world=haircut_only(),
    script=(
        Beat(
            caller="Haircut Monday at ten.",
            responses=(
                use_tools(_check()),
                use_tools(_book(TEN, ("", ADA[1]))),
                say("Can I take your name for that?"),
            ),
        ),
        Beat(
            caller="Ada Lovelace.",
            responses=(use_tools(_book(TEN, ADA)), say("Booked for 10:00, Ada.")),
        ),
    ),
    expect=Expectation(
        task="book",
        expected_tools=(
            ExpectedTool("check_availability", {"service_name": HAIRCUT}),
            ExpectedTool(
                "book_appointment", {"customer_name": ""}, succeeds=False
            ),
            ExpectedTool("book_appointment", {"customer_name": ADA[0]}),
        ),
        allow_tool_failures=frozenset({"book_appointment"}),
        final_appointments=(AppointmentSpec(HAIRCUT, ADA[0], TEN),),
        expected_model_calls=5,
    ),
)


SCENARIOS = (
    BOOKING_AVAILABLE,
    BOOKING_CALLER_CHANGES_MIND,
    BOOKING_UNAVAILABLE_SLOT,
    BOOKING_SLOT_TAKEN_BETWEEN_CHECK_AND_BOOK,
    BOOKING_UNKNOWN_SERVICE,
    BOOKING_AMBIGUOUS_SERVICE,
    BOOKING_MISSING_DETAILS,
)

__all__ = ["SCENARIOS"]
