"""Admission, draining, and the lock that keeps two threads out of one session.

The guard is the interesting half. It exists because of two things that are
both ordinary caller behaviour rather than abuse: talking again before the
first reply is finished, and hanging up while a turn is still running. The
first puts two threads into one `Session`; the second closes that `Session`
while a thread nobody can stop is still inside it.
"""

import threading
import time

import pytest

from app.runtime import (
    CallAdmission,
    CallRefused,
    ConversationGuard,
    TurnRejected,
    close_when_idle,
    get_admission,
    reset_admission,
)


class _Conversation:
    """A conversation that records overlap, so a lock can be proven to work."""

    def __init__(self, hold: float = 0.0) -> None:
        self.hold = hold
        self.inside = 0
        self.overlapped = False
        self.calls: list[str] = []

    def send(self, caller_text: str) -> str:
        self.inside += 1
        self.overlapped = self.overlapped or self.inside > 1
        try:
            if self.hold:
                time.sleep(self.hold)
            self.calls.append(caller_text)
            return f"reply to {caller_text}"
        finally:
            self.inside -= 1


class _Session:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


# --- admission -------------------------------------------------------------


def test_a_call_is_admitted_below_the_limit() -> None:
    admission = CallAdmission(limit=2)

    with admission.admit():
        assert admission.active == 1


def test_a_place_is_given_back_afterwards() -> None:
    admission = CallAdmission(limit=1)

    with admission.admit():
        pass

    assert admission.active == 0


def test_a_place_is_given_back_even_when_the_call_fails() -> None:
    admission = CallAdmission(limit=1)

    with pytest.raises(ValueError):
        with admission.admit():
            raise ValueError("the call went wrong")

    assert admission.active == 0


def test_the_call_past_the_limit_is_refused() -> None:
    """Refused immediately, not queued: a busy signal beats an endless wait."""
    admission = CallAdmission(limit=1)

    with admission.admit():
        with pytest.raises(CallRefused, match="at capacity"):
            with admission.admit():
                pass


def test_refusals_are_counted() -> None:
    admission = CallAdmission(limit=0)

    for _ in range(3):
        with pytest.raises(CallRefused):
            with admission.admit():
                pass

    assert admission.refused == 3


def test_a_draining_process_refuses_new_calls() -> None:
    admission = CallAdmission(limit=10)
    admission.start_draining()

    with pytest.raises(CallRefused, match="shutting down"):
        with admission.admit():
            pass


def test_draining_leaves_calls_already_in_progress_alone() -> None:
    admission = CallAdmission(limit=10)

    with admission.admit():
        admission.start_draining()
        assert admission.active == 1

    assert admission.active == 0


def test_the_process_wide_counter_follows_the_setting(settings_env) -> None:
    from app.config import get_settings

    reset_admission()

    assert get_admission().limit == get_settings().max_concurrent_calls


def test_the_process_wide_counter_is_one_object(settings_env) -> None:
    assert get_admission() is get_admission()


# --- the guard -------------------------------------------------------------


def test_the_guard_passes_a_turn_through() -> None:
    conversation = _Conversation()
    guard = ConversationGuard(conversation=conversation, timeout=1.0)

    assert guard.send("hello") == "reply to hello"
    assert conversation.calls == ["hello"]


def test_two_threads_never_enter_the_conversation_at_once() -> None:
    """The whole point: a `Session` is not thread-safe."""
    conversation = _Conversation(hold=0.05)
    guard = ConversationGuard(conversation=conversation, timeout=5.0)

    threads = [
        threading.Thread(target=guard.send, args=(f"turn {index}",))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert conversation.overlapped is False
    assert len(conversation.calls) == 4


def test_a_turn_that_waits_too_long_is_rejected_rather_than_queued() -> None:
    """Bounded, so a stuck turn cannot accumulate threads behind it."""
    conversation = _Conversation(hold=0.5)
    guard = ConversationGuard(conversation=conversation, timeout=0.05)

    blocker = threading.Thread(target=guard.send, args=("first",))
    blocker.start()
    time.sleep(0.05)
    try:
        with pytest.raises(TurnRejected):
            guard.send("second")
    finally:
        blocker.join()


def test_the_guard_reports_whether_anybody_is_inside() -> None:
    conversation = _Conversation(hold=0.2)
    guard = ConversationGuard(conversation=conversation, timeout=5.0)

    assert guard.busy is False
    thread = threading.Thread(target=guard.send, args=("x",))
    thread.start()
    time.sleep(0.05)
    try:
        assert guard.busy is True
    finally:
        thread.join()
    assert guard.busy is False


def test_a_failing_turn_still_releases_the_guard() -> None:
    class _Broken:
        def send(self, caller_text: str):
            raise RuntimeError("the model fell over")

    guard = ConversationGuard(conversation=_Broken(), timeout=1.0)

    with pytest.raises(RuntimeError):
        guard.send("x")

    assert guard.busy is False


# --- closing a session safely ----------------------------------------------


def test_an_idle_session_is_closed() -> None:
    session = _Session()
    guard = ConversationGuard(conversation=_Conversation(), timeout=1.0)

    assert close_when_idle(guard, session, grace=0.5) is True
    assert session.closed is True


def test_a_session_with_no_guard_is_closed() -> None:
    session = _Session()

    assert close_when_idle(None, session, grace=0.5) is True
    assert session.closed is True


def test_a_session_still_in_use_is_not_closed_underneath_the_thread() -> None:
    """The honest outcome: a leaked connection beats a corrupted session."""
    conversation = _Conversation(hold=0.6)
    guard = ConversationGuard(conversation=conversation, timeout=5.0)
    session = _Session()

    thread = threading.Thread(target=guard.send, args=("x",))
    thread.start()
    time.sleep(0.05)
    try:
        assert close_when_idle(guard, session, grace=0.05) is False
        assert session.closed is False
    finally:
        thread.join()


def test_closing_waits_for_a_turn_that_finishes_in_time() -> None:
    conversation = _Conversation(hold=0.1)
    guard = ConversationGuard(conversation=conversation, timeout=5.0)
    session = _Session()

    thread = threading.Thread(target=guard.send, args=("x",))
    thread.start()
    try:
        assert close_when_idle(guard, session, grace=5.0) is True
    finally:
        thread.join()
    assert session.closed is True


def test_nothing_happens_without_a_session() -> None:
    assert close_when_idle(None, None, grace=0.1) is False
