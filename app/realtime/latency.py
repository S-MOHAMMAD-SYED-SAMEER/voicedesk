"""Writing down what only the audio layer could have measured.

The dialogue layer records what was said and how long the model took. It
cannot know how long the caller spoke, how long recognition took after they
stopped, or when the first audio of the reply existed — it is finished before
any of that is true. So it returns the ids of the two rows it wrote, and this
fills in the three columns that have been waiting since milestone 1.

No new column, no new table, no migration. `playback_end` is deliberately not
stored: it is held in session state and used to derive a turn's length, and a
column for a number the carrier may never report would be worse than deriving
one.

Best effort throughout. A call is not worth failing over a metric.
"""

import logging
from typing import Protocol

from sqlalchemy.orm import Session

from app.models import Turn

logger = logging.getLogger(__name__)


class HasTurnIds(Protocol):
    """The part of a dialogue result this module needs."""

    caller_turn_id: object
    agent_turn_id: object


class HasTiming(Protocol):
    """The part of a turn's timing this module needs."""

    @property
    def audio_ms(self) -> int | None: ...

    @property
    def stt_latency_ms(self) -> int | None: ...

    @property
    def tts_latency_ms(self) -> int | None: ...


def record(session: Session, result: HasTurnIds, timing: HasTiming) -> None:
    """Attach measured latency to the rows the dialogue layer already wrote."""
    try:
        _write(session, result, timing)
    except Exception:  # noqa: BLE001 - metrics must not break a call
        logger.exception("Could not record latency for a turn.")
        session.rollback()


def _write(session: Session, result: HasTurnIds, timing: HasTiming) -> None:
    caller = (
        session.get(Turn, result.caller_turn_id) if result.caller_turn_id else None
    )
    agent = session.get(Turn, result.agent_turn_id) if result.agent_turn_id else None

    # How long the caller spoke belongs to the caller's row; how long the
    # system took to answer belongs to the answer's.
    if caller is not None and timing.audio_ms is not None:
        caller.audio_ms = timing.audio_ms
    if agent is not None:
        if timing.stt_latency_ms is not None:
            agent.stt_latency_ms = timing.stt_latency_ms
        if timing.tts_latency_ms is not None:
            agent.tts_latency_ms = timing.tts_latency_ms

    if caller is not None or agent is not None:
        session.commit()
