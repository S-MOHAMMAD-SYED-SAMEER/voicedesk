"""Moving and cancelling appointments that already exist.

`reschedule_success` is where the frozen gap shows, and it is deliberately not
papered over. `ToolExecutor` guards `book_appointment` against times the
calendar never offered; it does not guard `reschedule`, and the milestone-4
prompt asks only that a *booking* be checked first. So a move that nothing
verified is the system working as built — the appointment really does move,
the exclusion constraint really does prevent an overlap, and nothing looked at
whether the caller had been offered that time.

The scenario therefore **passes** and carries one non-blocking
`UNVERIFIED_RESCHEDULE` observation. That line in the report is the honest
picture of a real gap, which is worth more than a dataset arranged to avoid it.
"""

from app.evals.dataset.common import ADA, BEA, HAIRCUT, TEN, TWO
from app.evals.model import say, use_tools
from app.evals.scenario import (
    AppointmentSpec,
    Beat,
    ExpectedTool,
    Expectation,
    Scenario,
    SeedAppointment,
    ServiceSpec,
    World,
    weekdays,
)

EXISTING = "{{appointment:existing}}"


def _with(*seeds: SeedAppointment) -> World:
    return World(
        services=(ServiceSpec(HAIRCUT, 30, "sam"),),
        hours=weekdays(),
        appointments=seeds,
    )


RESCHEDULE_SUCCESS = Scenario(
    name="reschedule_success",
    kind="text",
    description=(
        "An existing booking is moved to a free time. Nothing verified that "
        "time first, because nothing in the system requires it to — recorded "
        "as an observation, not a failure."
    ),
    world=_with(SeedAppointment("existing", HAIRCUT, ADA[0], ADA[1], TEN)),
    script=(
        Beat(
            caller="Could we move my haircut to the afternoon?",
            responses=(
                use_tools(
                    ("reschedule", {"appointment_id": EXISTING, "new_starts_at": TWO})
                ),
                # No numeral: this time was never offered, and a reply naming
                # it would be indistinguishable to the claim detector from one
                # offering it.
                say("That's moved to the afternoon for you."),
            ),
        ),
    ),
    expect=Expectation(
        task="reschedule",
        expected_tools=(ExpectedTool("reschedule", {"new_starts_at": TWO}),),
        forbidden_tools=frozenset({"book_appointment"}),
        final_appointments=(AppointmentSpec(HAIRCUT, ADA[0], TWO),),
        expected_model_calls=2,
    ),
)


RESCHEDULE_CONFLICT = Scenario(
    name="reschedule_conflict",
    kind="text",
    description=(
        "A move straight onto a slot somebody else holds. The database's "
        "exclusion constraint refuses it and nothing changes."
    ),
    world=_with(
        SeedAppointment("existing", HAIRCUT, ADA[0], ADA[1], TEN),
        SeedAppointment("other", HAIRCUT, BEA[0], BEA[1], TWO),
    ),
    script=(
        Beat(
            caller="Move my haircut to two o'clock.",
            responses=(
                use_tools(
                    ("reschedule", {"appointment_id": EXISTING, "new_starts_at": TWO})
                ),
                say("Somebody has that time, I'm afraid. Shall I see what's left?"),
            ),
        ),
    ),
    expect=Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool("reschedule", {"new_starts_at": TWO}, succeeds=False),
        ),
        allow_tool_failures=frozenset({"reschedule"}),
        final_appointments=(
            AppointmentSpec(HAIRCUT, ADA[0], TEN),
            AppointmentSpec(HAIRCUT, BEA[0], TWO),
        ),
        expected_model_calls=2,
    ),
)


CANCEL_SUCCESS = Scenario(
    name="cancel_success",
    kind="text",
    description="An existing booking is cancelled, and the row is kept as cancelled.",
    world=_with(SeedAppointment("existing", HAIRCUT, ADA[0], ADA[1], TEN)),
    script=(
        Beat(
            caller="I need to cancel my haircut on Monday.",
            responses=(
                use_tools(("cancel", {"appointment_id": EXISTING})),
                say("That's cancelled. Ring us any time to rebook."),
            ),
        ),
    ),
    expect=Expectation(
        task="cancel",
        expected_tools=(ExpectedTool("cancel", {"appointment_id": EXISTING}),),
        forbidden_tools=frozenset({"book_appointment"}),
        final_appointments=(
            AppointmentSpec(HAIRCUT, ADA[0], TEN, status="cancelled"),
        ),
        expected_model_calls=2,
    ),
)


CANCEL_ALREADY_CANCELLED = Scenario(
    name="cancel_already_cancelled",
    kind="text",
    description=(
        "Cancelling something already cancelled. The tool refuses, and the "
        "caller is told plainly rather than told it worked."
    ),
    world=_with(
        SeedAppointment("existing", HAIRCUT, ADA[0], ADA[1], TEN, cancelled=True)
    ),
    script=(
        Beat(
            caller="Can you cancel my haircut?",
            responses=(
                use_tools(("cancel", {"appointment_id": EXISTING})),
                say("That one is already cancelled — there's nothing booked."),
            ),
        ),
    ),
    expect=Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool("cancel", {"appointment_id": EXISTING}, succeeds=False),
        ),
        allow_tool_failures=frozenset({"cancel"}),
        final_appointments=(
            AppointmentSpec(HAIRCUT, ADA[0], TEN, status="cancelled"),
        ),
        expected_model_calls=2,
    ),
)


SCENARIOS = (
    RESCHEDULE_SUCCESS,
    RESCHEDULE_CONFLICT,
    CANCEL_SUCCESS,
    CANCEL_ALREADY_CANCELLED,
)

__all__ = ["SCENARIOS"]
