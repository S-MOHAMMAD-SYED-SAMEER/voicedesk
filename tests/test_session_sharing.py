"""One call, one `Session`, one thread at a time.

A call's `Session` is not thread-safe, and more than a dialogue turn uses it:
when a turn ends the transport writes its latency and cost, and when the call
ends it records that. Those writes used to run on the event-loop thread without
taking the conversation guard, so a turn still inside `Conversation.send` on a
worker thread could be mid-commit on the very `Session` the loop thread then
committed — which SQLAlchemy refuses with `IllegalStateChangeError`.

That only happened when two turns overlapped or a call ended while a turn was
still running, so it showed up as a test that failed one full run in three. These
tests do not depend on timing: a turn is held part-way through a commit by a
hook, and the code under test is asked to do its write while it is held. It must
wait for the turn rather than go into the session beside it.
"""

import logging
import threading

import anyio
import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Call, Turn
from app.realtime.session import RealtimeTurn, TurnTiming
from app.runtime import ConversationGuard, close_when_idle, write_when_idle
from app.telephony.stream import StreamState, _record_latency

from .conftest import say


class TurnInsideTheSession:
    """A conversation whose turn is stopped part-way through a commit.

    Stands in for a model request and its tool calls on a worker thread: it has
    the session in the middle of an operation and does not give it back until
    the test lets it go.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self.inside = threading.Event()
        self.release = threading.Event()

    def send(self, caller_text: str) -> str:
        def hold(_session: Session) -> None:
            self.inside.set()
            assert self.release.wait(10), "the test never let the turn go"

        event.listen(self._session, "before_commit", hold, once=True)
        self._session.execute(text("SELECT 1"))
        self._session.commit()
        return "done"


def _start_turn(guard: ConversationGuard) -> threading.Thread:
    worker = threading.Thread(target=guard.send, args=("hello",))
    worker.start()
    return worker


def test_the_end_of_a_call_waits_for_a_turn_that_is_still_inside_the_session(
    session: Session, call: Call, migrated_engine, caplog
) -> None:
    call_id = call.id
    turn = TurnInsideTheSession(session)
    guard = ConversationGuard(conversation=turn, timeout=10)
    state = StreamState(settings=Settings(_env_file=None, shutdown_grace_seconds=10))
    state.session, state.call, state.guard = session, call, guard

    raised: list[BaseException] = []

    def close() -> None:
        try:
            state.close()
        except BaseException as exc:  # noqa: BLE001 - recorded, then asserted on
            raised.append(exc)

    worker = _start_turn(guard)
    closer = threading.Thread(target=close)
    try:
        assert turn.inside.wait(5)
        closer.start()
        closer.join(0.5)
        went_in_beside_the_turn = not closer.is_alive()
    finally:
        # Always let the turn go, or a failure here leaves a transaction open
        # and the fixture's teardown waits on it for ever.
        turn.release.set()
        worker.join(10)
        closer.join(10)

    assert not went_in_beside_the_turn, "close() used the session while a turn was inside it"
    assert raised == [], f"closing the call raised {raised!r}"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "closing the call logged an error"
    )
    with Session(migrated_engine) as fresh:
        assert fresh.get(Call, call_id).ended_at is not None


@pytest.mark.anyio
async def test_a_turns_latency_waits_for_a_turn_that_is_still_inside_the_session(
    session: Session, call: Call, dialogue, migrated_engine, caplog
) -> None:
    result = dialogue(say("Hello there.")).send("Hi")
    caller_turn_id = result.caller_turn_id

    turn = TurnInsideTheSession(session)
    guard = ConversationGuard(conversation=turn, timeout=10)
    state = StreamState(settings=Settings(_env_file=None))
    state.session, state.call, state.guard = session, call, guard

    finished = anyio.Event()
    heard = RealtimeTurn(
        generation=1,
        dialogue=result,
        timing=TurnTiming(speech_start=0.0, speech_end=1500.0),
    )

    async def record() -> None:
        try:
            await _record_latency(state, heard)
        finally:
            finished.set()

    worker = _start_turn(guard)
    try:
        async with anyio.create_task_group() as group:
            assert await anyio.to_thread.run_sync(turn.inside.wait, 5)

            group.start_soon(record)
            await anyio.sleep(0.3)
            went_in_beside_the_turn = finished.is_set()

            turn.release.set()
            await finished.wait()
    finally:
        # Always let the turn go; see the end-of-call test.
        turn.release.set()
        worker.join(10)

    assert not went_in_beside_the_turn, "the latency write used the session during a turn"
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "recording the latency logged an error"
    )
    with Session(migrated_engine) as fresh:
        assert fresh.get(Turn, caller_turn_id).audio_ms == 1500


def test_the_last_write_runs_only_once_nobody_is_inside_the_session(
    session: Session,
) -> None:
    turn = TurnInsideTheSession(session)
    guard = ConversationGuard(conversation=turn, timeout=10)
    finalised = threading.Event()

    worker = _start_turn(guard)
    assert turn.inside.wait(5)

    closer = threading.Thread(
        target=close_when_idle, args=(guard, session, 10, finalised.set)
    )
    closer.start()
    closer.join(0.3)
    ran_early = finalised.is_set()

    turn.release.set()
    worker.join(10)
    closer.join(10)

    assert not ran_early
    assert finalised.is_set()


def test_the_last_write_is_skipped_when_a_turn_never_lets_go(session: Session) -> None:
    turn = TurnInsideTheSession(session)
    guard = ConversationGuard(conversation=turn, timeout=10)
    finalised = threading.Event()

    worker = _start_turn(guard)
    assert turn.inside.wait(5)

    closed = close_when_idle(guard, session, 0.1, finalised.set)

    turn.release.set()
    worker.join(10)

    assert closed is False
    assert not finalised.is_set()


@pytest.mark.anyio
async def test_a_write_that_cannot_get_the_session_is_declined_not_forced(
    session: Session,
) -> None:
    turn = TurnInsideTheSession(session)
    guard = ConversationGuard(conversation=turn, timeout=0.1)
    ran = threading.Event()

    worker = _start_turn(guard)
    assert await anyio.to_thread.run_sync(turn.inside.wait, 5)

    wrote = await write_when_idle(guard, ran.set)

    turn.release.set()
    worker.join(10)

    assert wrote is False
    assert not ran.is_set()
