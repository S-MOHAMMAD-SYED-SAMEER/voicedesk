"""Process-wide limits, and the two safety rails a live call needs.

Three things live here, all of them transport concerns. None of them changes
how a conversation works, what the detector hears, or when a reply is cut off.

**Admission.** A call holds one database connection for its whole length, so
the number of calls a process will take has to be a number somebody chose
rather than whatever the connection pool happens to run out at. Past the
limit a call is refused immediately; a caller who hears a busy signal is
better served than one who waits on a connection that is not coming.

**Draining.** On shutdown the process stops accepting calls but lets the ones
in progress finish. There is no queue and no supervisor: a boolean, a count,
and a bounded wait.

**The conversation guard.** Two problems, one lock.

* `RealtimeSession` starts each turn in its own task, and every turn reaches
  the same `Session` through the same `Conversation`. A `Session` is not
  thread-safe, and two turns can overlap — a caller who talks again while the
  first reply is still being worked on is ordinary behaviour, not abuse.
* `anyio.to_thread.run_sync(..., abandon_on_cancel=True)` returns the moment a
  turn is cancelled, but the thread it abandoned keeps running inside
  `Conversation.send`. If the transport then closes the session — which is
  exactly what hanging up does — that thread is using a closed `Session` from
  another thread.

The guard solves both by letting one thread at a time into the conversation,
and by giving teardown a way to wait until nobody is inside it. Where waiting
is not enough it says so and declines to close, because a leaked session that
is collected later is a smaller problem than a closed one still in use.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class CallRefused(RuntimeError):
    """This process will not take another call right now."""


class TurnRejected(RuntimeError):
    """A turn waited too long for the conversation and gave up.

    Raised from the guard rather than from any provider, and handled by the
    same path a failed turn already takes: the caller is told something safe
    and the call goes on.
    """


# --- admission and draining -------------------------------------------------


@dataclass
class CallAdmission:
    """How many calls this process is taking, and whether it is still taking any.

    Touched only from the event loop, so a plain counter is enough — there is
    no lock here because there is nothing to race with.
    """

    limit: int
    active: int = 0
    draining: bool = False
    refused: int = 0

    @property
    def full(self) -> bool:
        return self.active >= self.limit

    @property
    def accepting(self) -> bool:
        return not self.draining and not self.full

    def start_draining(self) -> None:
        """Stop taking new calls. The ones in progress are left alone."""
        self.draining = True

    def refuse_reason(self) -> str:
        if self.draining:
            return "shutting down"
        return "at capacity"

    @contextmanager
    def admit(self) -> Iterator[None]:
        """Hold a place for one call, or refuse it before anything is built."""
        if not self.accepting:
            self.refused += 1
            raise CallRefused(self.refuse_reason())
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1


@lru_cache
def get_admission() -> CallAdmission:
    """The one admission counter this process uses."""
    return CallAdmission(limit=get_settings().max_concurrent_calls)


def reset_admission() -> None:
    """Drop the cached counter. For tests, and for settings that changed."""
    get_admission.cache_clear()


# --- the conversation guard -------------------------------------------------


@dataclass
class ConversationGuard:
    """One conversation, entered by one thread at a time.

    Duck-typed on purpose: it exposes `send` and nothing else, which is all
    either session object asks of a conversation. Neither `VoiceSession` nor
    `RealtimeSession` can tell the difference, and neither was changed.
    """

    conversation: object
    timeout: float
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def send(self, caller_text: str):
        """One turn, once the turn before it has finished with the database."""
        if not self._lock.acquire(timeout=self.timeout):
            raise TurnRejected(
                f"A turn waited {self.timeout:g}s for the previous one to "
                "finish and gave up."
            )
        try:
            return self.conversation.send(caller_text)
        finally:
            self._lock.release()

    @property
    def busy(self) -> bool:
        """Is a thread inside the conversation right now?"""
        if self._lock.acquire(blocking=False):
            self._lock.release()
            return False
        return True

    def claim(self, timeout: float) -> bool:
        """Wait until nobody is inside, and keep it that way.

        Returns whether the wait succeeded. A caller that gets `True` owns the
        conversation and may close the session under it; a caller that gets
        `False` must not, because a thread it cannot stop is still in there.
        """
        return self._lock.acquire(timeout=timeout)


def close_when_idle(guard: ConversationGuard | None, session, grace: float) -> bool:
    """Close a call's session once no thread can still be using it.

    Returns whether it was closed. When a turn is still running — an
    abandoned model request, most likely, which cannot be cancelled and
    cannot be killed — the session is deliberately left open and the
    connection is returned when it is collected. That wastes a connection for
    a while. Closing it while another thread holds it would corrupt the
    session's state and the pool's, which is worse.
    """
    if session is None:
        return False
    if guard is None or guard.claim(grace):
        session.close()
        return True

    logger.error(
        "A turn was still running after %.1fs, so its database session was "
        "left for the collector rather than closed underneath it. A model "
        "request on a worker thread cannot be cancelled once it has started.",
        grace,
    )
    return False


__all__ = [
    "CallAdmission",
    "CallRefused",
    "ConversationGuard",
    "TurnRejected",
    "close_when_idle",
    "get_admission",
    "reset_admission",
]
