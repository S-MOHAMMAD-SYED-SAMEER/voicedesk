"""The two scenarios that cannot be run at the dialogue layer.

**Silence.** `Conversation.send("")` would still ask the model — it is
`RealtimeSession` that declines to, above it. So proving "silence asks no
model" means feeding real frames through the real voice-activity detector and
watching nothing happen. This scenario carries **no script at all**: if
anything reaches the model it runs out of responses immediately and the
scenario fails with a scenario-authoring finding, which is the assertion.

**Barge-in.** Entirely a property of milestone 7's generation model. The
booking is committed before the caller interrupts, and must stay committed:
cancellation is partial by nature — a thread running a model request and the
tools it called cannot be killed, only abandoned — so the guarantee being
measured is not "the work stopped" but "nothing stale was heard and nothing
was done twice".

Neither scenario changes anything in milestone 7. They watch it from outside,
through the session's own counters and the sink's own record.
"""

from app.evals.dataset.common import ADA, HAIRCUT, MONDAY, TEN, haircut_only
from app.evals.model import say, use_tools
from app.evals.scenario import (
    AppointmentSpec,
    Beat,
    Claim,
    ExpectedTool,
    Expectation,
    RealtimeExpectation,
    Scenario,
)
from app.tools import TOOLS

# The offline streaming transcriber returns one fixed sentence whatever it is
# given, so the caller's words are the scenario's documentation rather than
# its input. The frames are what drive this call.
SPOKEN = "I'd like to book an appointment"


REALTIME_BARGE_IN_BOOKING_SURVIVES = Scenario(
    name="realtime_barge_in_booking_survives",
    kind="realtime",
    description=(
        "A booking completes, the reply begins playing, and the caller talks "
        "over it. The reply is cut off and discarded; the booking stands."
    ),
    world=haircut_only(),
    script=(
        Beat(
            caller=SPOKEN,
            responses=(
                use_tools(
                    (
                        "check_availability",
                        {"service_name": HAIRCUT, "day": MONDAY},
                    )
                ),
                use_tools(
                    (
                        "book_appointment",
                        {
                            "service_name": HAIRCUT,
                            "starts_at": TEN,
                            "customer_name": ADA[0],
                            "phone": ADA[1],
                        },
                    )
                ),
                # Long enough that there is still some of it left to interrupt.
                say(
                    "You're booked in for 10:00 on Monday, Ada. Is there "
                    "anything else at all I can help you with today?"
                ),
            ),
        ),
    ),
    speech_frames=10,
    silence_frames=40,
    interrupt=True,
    expect=Expectation(
        task="book",
        expected_tools=(
            ExpectedTool("check_availability", {"service_name": HAIRCUT}),
            ExpectedTool("book_appointment", {"starts_at": TEN}),
        ),
        availability_claims=(Claim(HAIRCUT, TEN, 0),),
        # Committed before the interruption, and still there afterwards.
        final_appointments=(AppointmentSpec(HAIRCUT, ADA[0], TEN),),
        # Three requests for one turn: check, book, speak. A fourth would mean
        # the interruption had caused the turn to be run again.
        expected_model_calls=3,
        realtime=RealtimeExpectation(
            turns=1,
            interrupted=True,
            generation_advances=True,
            audio_discarded=True,
        ),
    ),
)


REALTIME_SILENCE_ASKS_NO_MODEL = Scenario(
    name="realtime_silence_asks_no_model",
    kind="realtime",
    description=(
        "Nothing but silence on the line. No turn, no model request, no tool "
        "call, and nothing invented to fill the gap."
    ),
    world=haircut_only(),
    # Deliberately empty. Any model request at all exhausts the script and
    # fails the scenario, which is exactly the assertion being made.
    script=(),
    speech_frames=0,
    silence_frames=60,
    interrupt=False,
    expect=Expectation(
        task="decline",
        expected_tools=(),
        forbidden_tools=frozenset(TOOLS),
        final_appointments=(),
        expected_model_calls=0,
        realtime=RealtimeExpectation(turns=0),
    ),
)


SCENARIOS = (
    REALTIME_BARGE_IN_BOOKING_SURVIVES,
    REALTIME_SILENCE_ASKS_NO_MODEL,
)

__all__ = ["SCENARIOS"]
