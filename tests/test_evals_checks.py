"""Every failure category, in both directions, and the three hallucination layers.

These are pure: a trace is built by hand and judged. No database, no model,
no call. That is what makes it possible to test a category that the dataset
never provokes — a suite that could only assert what its own scenarios happen
to produce would be measuring itself.
"""

from datetime import time

import pytest

from app.dialogue import ToolCallRecord
from app.evals.checks import (
    BOOKING_FAILURE,
    CANCELLATION_FAILURE,
    DUPLICATE_ACTION,
    HALLUCINATED_AVAILABILITY,
    INVALID_TOOL_ARGUMENT,
    MISSED_ESCALATION,
    MISSING_TOOL,
    NON_BLOCKING,
    PROVIDER_FAILURE,
    RESCHEDULE_FAILURE,
    TURN_LIMIT,
    UNDECLARED_CLAIM,
    UNEXPECTED_ACTION,
    UNNECESSARY_ESCALATION,
    UNVERIFIED_RESCHEDULE,
    WRONG_TOOL,
    assess,
    clock_times,
)
from app.evals.model import say
from app.evals.scenario import (
    AppointmentSpec,
    Beat,
    Claim,
    ExpectedTool,
    Expectation,
    RealtimeExpectation,
    Scenario,
    ServiceSpec,
    World,
    weekdays,
)
from app.evals.trace import SCRIPT_EXHAUSTED, CallTrace, TurnTrace

MONDAY = "2026-03-02"
NINE = "2026-03-02T09:00:00+00:00"
TEN = "2026-03-02T10:00:00+00:00"
ELEVEN = "2026-03-02T11:00:00+00:00"

HAIRCUT = "Haircut"
ADA = "Ada Lovelace"


# --- building traces by hand -----------------------------------------------


def _scenario(expect: Expectation, *, turns: int = 4, kind: str = "text") -> Scenario:
    return Scenario(
        name="example",
        kind=kind,
        description="An example.",
        world=World(services=(ServiceSpec(HAIRCUT),), hours=weekdays()),
        script=tuple(
            Beat(caller="x", responses=(say("y"),)) for _ in range(max(turns, 1))
        ),
        expect=expect,
        max_turns=turns,
    )


def _record(name: str, arguments: dict, *, success: bool = True, **data) -> ToolCallRecord:
    return ToolCallRecord(
        tool_name=name,
        arguments=arguments,
        success=success,
        data=data,
        error=None if success else "it did not work",
    )


def _offer(*starts: str, service: str = HAIRCUT) -> ToolCallRecord:
    return _record(
        "check_availability",
        {"service_name": service, "day": MONDAY},
        service=service,
        slots=[{"starts_at": moment, "ends_at": moment} for moment in starts],
    )


def _trace(*turns: TurnTrace, appointments: tuple = ()) -> CallTrace:
    trace = CallTrace(scenario="example", kind="text")
    trace.turns = list(turns)
    trace.final_appointments = list(appointments)
    return trace


def _turn(
    index: int = 0,
    *,
    reply: str = "",
    tools: tuple[ToolCallRecord, ...] = (),
    escalated: bool = False,
    booked: str | None = None,
    failed: bool = False,
) -> TurnTrace:
    return TurnTrace(
        index=index,
        caller_text="x",
        reply=reply,
        tool_calls=list(tools),
        escalated=escalated,
        booked_appointment_id=booked,
        failed=failed,
    )


def _categories(assessment) -> list[str]:
    return [finding.category for finding in assessment.findings]


# --- a clean pass ----------------------------------------------------------


def test_a_scenario_that_did_everything_right_has_no_findings() -> None:
    expect = Expectation(
        task="book",
        expected_tools=(
            ExpectedTool("check_availability", {"service_name": HAIRCUT}),
            ExpectedTool("book_appointment", {"starts_at": TEN}),
        ),
        final_appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),),
    )
    trace = _trace(
        _turn(tools=(_offer(TEN), _record("book_appointment", {"service_name": HAIRCUT, "starts_at": TEN}))),
        appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),),
    )

    assessment = assess(_scenario(expect), trace)

    assert assessment.findings == []
    assert assessment.passed is True
    assert assessment.tools.correct == 2


# --- tool categories -------------------------------------------------------


def test_a_forbidden_tool_is_a_wrong_tool() -> None:
    expect = Expectation(task="decline", forbidden_tools=frozenset({"book_appointment"}))
    trace = _trace(_turn(tools=(_record("book_appointment", {"starts_at": TEN}),)))

    assessment = assess(_scenario(expect), trace)

    assert WRONG_TOOL in _categories(assessment)
    assert assessment.tools.wrong == 1


def test_an_expected_tool_that_never_ran_is_missing() -> None:
    expect = Expectation(
        task="decline", expected_tools=(ExpectedTool("take_message"),)
    )

    assessment = assess(_scenario(expect), _trace(_turn()))

    assert MISSING_TOOL in _categories(assessment)
    assert assessment.tools.missing == 1


def test_the_right_tool_with_the_wrong_arguments_is_an_invalid_argument() -> None:
    expect = Expectation(
        task="decline",
        expected_tools=(ExpectedTool("check_availability", {"day": MONDAY}),),
    )
    trace = _trace(_turn(tools=(_record("check_availability", {"day": "2026-03-09"}),)))

    assessment = assess(_scenario(expect), trace)

    assert INVALID_TOOL_ARGUMENT in _categories(assessment)
    assert assessment.tools.invalid_argument == 1


def test_the_same_successful_call_twice_is_a_duplicate() -> None:
    expect = Expectation(
        task="decline",
        expected_tools=(ExpectedTool("take_message", {"caller_name": ADA}),),
    )
    call = _record("take_message", {"caller_name": ADA})
    trace = _trace(_turn(tools=(call, call)))

    assessment = assess(_scenario(expect), trace)

    assert DUPLICATE_ACTION in _categories(assessment)
    assert assessment.tools.duplicate == 1


def test_a_tool_nobody_expected_is_an_unexpected_action() -> None:
    expect = Expectation(task="decline")
    trace = _trace(_turn(tools=(_record("take_message", {"caller_name": ADA}),)))

    assessment = assess(_scenario(expect), trace)

    assert UNEXPECTED_ACTION in _categories(assessment)


def test_a_tool_expected_to_fail_and_failing_is_correct() -> None:
    expect = Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool("book_appointment", {"starts_at": TEN}, succeeds=False),
        ),
        allow_tool_failures=frozenset({"book_appointment"}),
    )
    trace = _trace(
        _turn(tools=(_record("book_appointment", {"starts_at": TEN}, success=False),))
    )

    assessment = assess(_scenario(expect), trace)

    assert assessment.findings == []
    assert assessment.tools.correct == 1


def test_a_tool_expected_to_fail_but_succeeding_does_not_match() -> None:
    """Succeeding where the scenario said it would fail is not the same call."""
    expect = Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool("book_appointment", {"starts_at": TEN}, succeeds=False),
        ),
    )
    trace = _trace(_turn(tools=(_record("book_appointment", {"starts_at": TEN}),)))

    assessment = assess(_scenario(expect), trace)

    assert MISSING_TOOL in _categories(assessment)


# --- escalation ------------------------------------------------------------


def test_not_escalating_when_required_is_a_missed_escalation() -> None:
    expect = Expectation(task="escalate", should_escalate=True)

    assessment = assess(_scenario(expect), _trace(_turn()))

    assert MISSED_ESCALATION in _categories(assessment)


def test_escalating_when_not_required_is_unnecessary() -> None:
    expect = Expectation(task="book")
    trace = _trace(
        _turn(
            escalated=True,
            tools=(_record("transfer_to_human", {"reason": "r"}, reason="r"),),
        )
    )

    assessment = assess(_scenario(expect), trace)

    assert UNNECESSARY_ESCALATION in _categories(assessment)


def test_escalating_when_required_is_no_finding() -> None:
    expect = Expectation(
        task="escalate",
        should_escalate=True,
        expected_tools=(ExpectedTool("transfer_to_human", {"reason": "r"}),),
    )
    trace = _trace(
        _turn(
            escalated=True,
            tools=(_record("transfer_to_human", {"reason": "r"}, reason="r"),),
        )
    )

    assert assess(_scenario(expect), trace).findings == []


# --- hallucination, layer one: structural ----------------------------------


def test_a_booking_at_a_time_never_offered_is_hallucinated() -> None:
    """The executor's guard should make this unreachable. It is asserted anyway."""
    expect = Expectation(
        task="book",
        expected_tools=(ExpectedTool("book_appointment", {"starts_at": ELEVEN}),),
        final_appointments=(AppointmentSpec(HAIRCUT, ADA, ELEVEN),),
    )
    trace = _trace(
        _turn(
            tools=(
                _offer(TEN),
                _record("book_appointment", {"service_name": HAIRCUT, "starts_at": ELEVEN}),
            )
        ),
        appointments=(AppointmentSpec(HAIRCUT, ADA, ELEVEN),),
    )

    assert HALLUCINATED_AVAILABILITY in _categories(assess(_scenario(expect), trace))


def test_a_booking_at_an_offered_time_is_not_hallucinated() -> None:
    expect = Expectation(
        task="book",
        expected_tools=(ExpectedTool("book_appointment", {"starts_at": TEN}),),
        final_appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),),
    )
    trace = _trace(
        _turn(
            tools=(
                _offer(TEN),
                _record("book_appointment", {"service_name": HAIRCUT, "starts_at": TEN}),
            )
        ),
        appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),),
    )

    assert HALLUCINATED_AVAILABILITY not in _categories(assess(_scenario(expect), trace))


def test_an_offer_of_one_service_does_not_verify_another() -> None:
    """The ledger is keyed by service and instant, exactly as the executor is."""
    expect = Expectation(
        task="book",
        expected_tools=(ExpectedTool("book_appointment", {"starts_at": TEN}),),
        final_appointments=(AppointmentSpec("Beard trim", ADA, TEN),),
    )
    trace = _trace(
        _turn(
            tools=(
                _offer(TEN, service=HAIRCUT),
                _record("book_appointment", {"service_name": "Beard trim", "starts_at": TEN}),
            )
        ),
        appointments=(AppointmentSpec("Beard trim", ADA, TEN),),
    )

    assert HALLUCINATED_AVAILABILITY in _categories(assess(_scenario(expect), trace))


def test_a_failed_booking_is_never_hallucinated() -> None:
    """Nothing happened, so nothing was promised."""
    expect = Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool("book_appointment", {"starts_at": ELEVEN}, succeeds=False),
        ),
        allow_tool_failures=frozenset({"book_appointment"}),
    )
    trace = _trace(
        _turn(
            tools=(
                _record(
                    "book_appointment",
                    {"service_name": HAIRCUT, "starts_at": ELEVEN},
                    success=False,
                ),
            )
        )
    )

    assert HALLUCINATED_AVAILABILITY not in _categories(assess(_scenario(expect), trace))


def test_an_unreadable_booking_time_is_treated_as_unverified() -> None:
    """Failing closed: a value nobody understood is not a verified slot."""
    expect = Expectation(
        task="book",
        expected_tools=(ExpectedTool("book_appointment"),),
    )
    trace = _trace(
        _turn(
            tools=(
                _offer(TEN),
                _record("book_appointment", {"service_name": HAIRCUT, "starts_at": "soon"}),
            )
        )
    )

    assert HALLUCINATED_AVAILABILITY in _categories(assess(_scenario(expect), trace))


# --- hallucination, layer two: declared claims -----------------------------


def test_a_claim_the_calendar_never_offered_is_hallucinated() -> None:
    expect = Expectation(
        task="decline", availability_claims=(Claim(HAIRCUT, ELEVEN, 0),)
    )
    trace = _trace(_turn(tools=(_offer(TEN),)))

    assert HALLUCINATED_AVAILABILITY in _categories(assess(_scenario(expect), trace))


def test_a_claim_the_calendar_did_offer_is_fine() -> None:
    expect = Expectation(
        task="decline",
        expected_tools=(ExpectedTool("check_availability", {"day": MONDAY}),),
        availability_claims=(Claim(HAIRCUT, TEN, 0),),
    )
    trace = _trace(_turn(tools=(_offer(TEN),)))

    assert assess(_scenario(expect), trace).findings == []


def test_a_claim_made_before_the_calendar_was_asked_is_hallucinated() -> None:
    """The ledger is a prefix. Offering it later does not excuse saying it first."""
    expect = Expectation(
        task="decline", availability_claims=(Claim(HAIRCUT, TEN, 0),)
    )
    trace = _trace(_turn(index=0), _turn(index=1, tools=(_offer(TEN),)))

    assert HALLUCINATED_AVAILABILITY in _categories(assess(_scenario(expect), trace))


def test_a_claim_on_a_later_turn_sees_the_earlier_ledger() -> None:
    expect = Expectation(
        task="decline", availability_claims=(Claim(HAIRCUT, TEN, 1),)
    )
    trace = _trace(_turn(index=0, tools=(_offer(TEN),)), _turn(index=1))

    assert HALLUCINATED_AVAILABILITY not in _categories(assess(_scenario(expect), trace))


def test_a_failed_availability_check_offers_nothing() -> None:
    expect = Expectation(
        task="decline", availability_claims=(Claim(HAIRCUT, TEN, 0),)
    )
    failed = _record(
        "check_availability",
        {"service_name": HAIRCUT, "day": MONDAY},
        success=False,
        service=HAIRCUT,
        slots=[{"starts_at": TEN, "ends_at": TEN}],
    )
    trace = _trace(_turn(tools=(failed,)))

    assert HALLUCINATED_AVAILABILITY in _categories(assess(_scenario(expect), trace))


# --- hallucination, layer three: the clock detector ------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I have 10:00 free.", {time(10)}),
        ("09:30 or 10:00?", {time(9, 30), time(10)}),
        ("How about 3pm?", {time(15)}),
        ("How about 3 pm?", {time(15)}),
        ("At 9am.", {time(9)}),
        ("12am is midnight.", {time(0)}),
        ("12pm is midday.", {time(12)}),
        ("10:30am works.", {time(10, 30)}),
        ("10:30 pm works.", {time(22, 30)}),
    ],
)
def test_the_detector_finds_unambiguous_clock_times(text, expected) -> None:
    assert clock_times(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I have nine, half nine or ten.",
        "Ring us on 07700 900123.",
        "We'll call you back within 24 hours.",
        "That's 30 minutes long.",
        "",
        "See you Monday.",
        "Version 2.5 of the system.",
    ],
)
def test_the_detector_stays_silent_on_anything_ambiguous(text) -> None:
    """Conservative on purpose: it must never fire on something it guessed."""
    assert clock_times(text) == set()


def test_a_time_in_a_reply_that_was_offered_is_accounted_for() -> None:
    expect = Expectation(task="decline")
    trace = _trace(_turn(reply="I have 10:00 free.", tools=(_offer(TEN),)))

    assert UNDECLARED_CLAIM not in _categories(assess(_scenario(expect), trace))


def test_a_time_in_a_reply_that_nothing_accounts_for_is_undeclared() -> None:
    """The scenario author wrote a time and forgot to declare it."""
    expect = Expectation(task="decline")
    trace = _trace(_turn(reply="How about 15:00?", tools=(_offer(TEN),)))

    assert UNDECLARED_CLAIM in _categories(assess(_scenario(expect), trace))


def test_a_declared_claim_accounts_for_a_time_in_a_reply() -> None:
    expect = Expectation(
        task="decline", availability_claims=(Claim(HAIRCUT, TEN, 0),)
    )
    trace = _trace(_turn(reply="I have 10:00 free.", tools=(_offer(TEN),)))

    assert UNDECLARED_CLAIM not in _categories(assess(_scenario(expect), trace))


def test_an_undeclared_claim_is_blocking() -> None:
    """A scenario that says more than its ground truth cannot be scored."""
    expect = Expectation(task="decline")
    trace = _trace(_turn(reply="How about 15:00?"))

    assessment = assess(_scenario(expect), trace)

    assert assessment.passed is False


def test_the_detector_only_reads_replies_not_caller_speech() -> None:
    """The caller may say any time they like; only the receptionist is judged."""
    expect = Expectation(task="decline")
    trace = _trace(_turn(reply=""))
    trace.turns[0].caller_text = "Can I have 15:00?"

    assert UNDECLARED_CLAIM not in _categories(assess(_scenario(expect), trace))


# --- the non-blocking reschedule observation -------------------------------


def test_a_reschedule_to_an_unoffered_time_is_observed() -> None:
    expect = Expectation(
        task="reschedule",
        expected_tools=(ExpectedTool("reschedule", {"new_starts_at": ELEVEN}),),
    )
    trace = _trace(
        _turn(tools=(_record("reschedule", {"appointment_id": "x", "new_starts_at": ELEVEN}),))
    )

    assessment = assess(_scenario(expect), trace)

    assert UNVERIFIED_RESCHEDULE in _categories(assessment)


def test_the_reschedule_observation_never_fails_a_scenario() -> None:
    """Frozen milestone-3/4 behaviour is measured, not punished."""
    expect = Expectation(
        task="reschedule",
        expected_tools=(ExpectedTool("reschedule", {"new_starts_at": ELEVEN}),),
    )
    trace = _trace(
        _turn(tools=(_record("reschedule", {"appointment_id": "x", "new_starts_at": ELEVEN}),))
    )

    assessment = assess(_scenario(expect), trace)

    assert assessment.passed is True
    assert [finding.category for finding in assessment.observations] == [
        UNVERIFIED_RESCHEDULE
    ]


def test_a_reschedule_to_an_offered_time_is_not_observed() -> None:
    expect = Expectation(
        task="reschedule",
        expected_tools=(
            ExpectedTool("check_availability"),
            ExpectedTool("reschedule", {"new_starts_at": TEN}),
        ),
    )
    trace = _trace(
        _turn(
            tools=(
                _offer(TEN),
                _record("reschedule", {"appointment_id": "x", "new_starts_at": TEN}),
            )
        )
    )

    assert UNVERIFIED_RESCHEDULE not in _categories(assess(_scenario(expect), trace))


def test_a_failed_reschedule_is_not_observed() -> None:
    expect = Expectation(
        task="decline",
        expected_tools=(
            ExpectedTool("reschedule", {"new_starts_at": ELEVEN}, succeeds=False),
        ),
        allow_tool_failures=frozenset({"reschedule"}),
    )
    trace = _trace(
        _turn(
            tools=(
                _record(
                    "reschedule",
                    {"appointment_id": "x", "new_starts_at": ELEVEN},
                    success=False,
                ),
            )
        )
    )

    assert UNVERIFIED_RESCHEDULE not in _categories(assess(_scenario(expect), trace))


def test_only_the_reschedule_observation_is_non_blocking() -> None:
    assert NON_BLOCKING == frozenset({UNVERIFIED_RESCHEDULE})


# --- final state -----------------------------------------------------------


def test_a_missing_booking_is_a_booking_failure() -> None:
    expect = Expectation(
        task="book", final_appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),)
    )

    assert BOOKING_FAILURE in _categories(assess(_scenario(expect), _trace(_turn())))


def test_a_missing_reschedule_is_a_reschedule_failure() -> None:
    expect = Expectation(
        task="reschedule", final_appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),)
    )

    assert RESCHEDULE_FAILURE in _categories(assess(_scenario(expect), _trace(_turn())))


def test_a_missing_cancellation_is_a_cancellation_failure() -> None:
    expect = Expectation(
        task="cancel",
        final_appointments=(AppointmentSpec(HAIRCUT, ADA, TEN, "cancelled"),),
    )
    trace = _trace(_turn(), appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),))

    assert CANCELLATION_FAILURE in _categories(assess(_scenario(expect), trace))


def test_an_appointment_nobody_expected_fails_a_decline() -> None:
    expect = Expectation(task="decline", final_appointments=())
    trace = _trace(_turn(), appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),))

    assert UNEXPECTED_ACTION in _categories(assess(_scenario(expect), trace))


def test_final_state_compares_instants_not_strings() -> None:
    expect = Expectation(
        task="book",
        final_appointments=(AppointmentSpec(HAIRCUT, ADA, "2026-03-02T11:00:00+01:00"),),
    )
    trace = _trace(_turn(), appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),))

    assert BOOKING_FAILURE not in _categories(assess(_scenario(expect), trace))


def test_final_state_compares_names_case_insensitively() -> None:
    expect = Expectation(
        task="book", final_appointments=(AppointmentSpec("haircut", "ada lovelace", TEN),)
    )
    trace = _trace(_turn(), appointments=(AppointmentSpec(HAIRCUT, ADA, TEN),))

    assert BOOKING_FAILURE not in _categories(assess(_scenario(expect), trace))


# --- the rest --------------------------------------------------------------


def test_too_many_turns_is_a_turn_limit() -> None:
    expect = Expectation(task="decline")
    trace = _trace(*[_turn(index) for index in range(5)])

    assert TURN_LIMIT in _categories(assess(_scenario(expect, turns=2), trace))


def test_a_failed_turn_is_a_provider_failure() -> None:
    expect = Expectation(task="decline")

    assert PROVIDER_FAILURE in _categories(
        assess(_scenario(expect), _trace(_turn(failed=True)))
    )


def test_a_call_that_never_ran_reports_only_that() -> None:
    """Nothing else can be judged honestly about a call that did not happen."""
    trace = _trace()
    trace.error, trace.error_category = "boom", PROVIDER_FAILURE

    assessment = assess(_scenario(Expectation(task="book")), trace)

    assert _categories(assessment) == [PROVIDER_FAILURE]


def test_an_exhausted_script_is_reported_as_an_authoring_fault() -> None:
    trace = _trace()
    trace.error, trace.error_category = "ran out", SCRIPT_EXHAUSTED

    assessment = assess(_scenario(Expectation(task="book")), trace)

    assert _categories(assessment) == [SCRIPT_EXHAUSTED]
    assert assessment.findings[0].authoring is True
    assert assessment.passed is False


def test_more_model_calls_than_expected_is_a_duplicate() -> None:
    expect = Expectation(task="decline", expected_model_calls=2)
    trace = _trace(_turn())
    trace.model_calls = 3

    assert DUPLICATE_ACTION in _categories(assess(_scenario(expect), trace))


def test_fewer_model_calls_than_expected_is_unexpected() -> None:
    expect = Expectation(task="decline", expected_model_calls=2)
    trace = _trace(_turn())
    trace.model_calls = 1

    assert UNEXPECTED_ACTION in _categories(assess(_scenario(expect), trace))


def test_the_model_call_count_is_not_asserted_unless_the_scenario_says() -> None:
    trace = _trace(_turn())
    trace.model_calls = 99

    assert assess(_scenario(Expectation(task="decline")), trace).findings == []


# --- realtime expectations -------------------------------------------------


def test_an_uninterrupted_turn_where_one_was_expected_is_a_finding() -> None:
    expect = Expectation(
        task="decline", realtime=RealtimeExpectation(interrupted=True)
    )

    assert UNEXPECTED_ACTION in _categories(
        assess(_scenario(expect, kind="realtime"), _trace(_turn()))
    )


def test_an_interrupted_turn_where_one_was_expected_is_fine() -> None:
    expect = Expectation(
        task="decline", realtime=RealtimeExpectation(interrupted=True)
    )
    trace = _trace(_turn())
    trace.turns[0].interrupted = True

    assert assess(_scenario(expect, kind="realtime"), trace).findings == []


def test_a_generation_that_did_not_advance_is_a_finding() -> None:
    expect = Expectation(
        task="decline", realtime=RealtimeExpectation(generation_advances=True)
    )
    trace = _trace(_turn())
    trace.generation = 1

    assert UNEXPECTED_ACTION in _categories(
        assess(_scenario(expect, kind="realtime"), trace)
    )


def test_a_generation_past_the_turn_count_has_advanced() -> None:
    expect = Expectation(
        task="decline", realtime=RealtimeExpectation(generation_advances=True)
    )
    trace = _trace(_turn())
    trace.generation = 2

    assert assess(_scenario(expect, kind="realtime"), trace).findings == []


def test_audio_merely_stopped_rather_than_discarded_is_a_finding() -> None:
    expect = Expectation(
        task="decline", realtime=RealtimeExpectation(audio_discarded=True)
    )
    trace = _trace(_turn())
    trace.sink_clears, trace.sink_chunks = 0, 3

    assert UNEXPECTED_ACTION in _categories(
        assess(_scenario(expect, kind="realtime"), trace)
    )


def test_audio_cleared_and_emptied_counts_as_discarded() -> None:
    expect = Expectation(
        task="decline", realtime=RealtimeExpectation(audio_discarded=True)
    )
    trace = _trace(_turn())
    trace.sink_clears, trace.sink_chunks = 1, 0

    assert assess(_scenario(expect, kind="realtime"), trace).findings == []


def test_a_turn_where_silence_was_expected_is_a_finding() -> None:
    expect = Expectation(task="decline", realtime=RealtimeExpectation(turns=0))

    assert UNEXPECTED_ACTION in _categories(
        assess(_scenario(expect, kind="realtime"), _trace(_turn()))
    )
