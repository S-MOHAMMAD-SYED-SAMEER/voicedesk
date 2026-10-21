"""The holding line: something to hear while the model is thinking.

A model request can take seconds, and a caller who hears nothing for that long assumes the line has dropped. So if the
work takes longer than a short delay, the caller hears a fixed line ("One moment, let me check.") once.

    result = await with_holding_line(work, say, after_seconds=2.0)

`work` runs; if it has not finished after `after_seconds`, `say()` runs once beside it. Three properties matter:

* **Said at most once** per call of this function, however long the work takes.
* **Never cut off.** Once the line has started it is allowed to finish, so the reply that follows is never talking
  over it; if the work finishes first, the line is never said at all.
* **Never part of the conversation.** It is synthesised and played, nothing more: not written to the transcript, not
  sent to the model. (Its characters are therefore not counted in the cost rows, a small known under-count, like the
  greeting's.)

If saying it fails, the failure is logged and the work carries on: a holding line is a courtesy, not a dependency.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import anyio

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_holding_line(
    work: Callable[[], Awaitable[T]],
    say: Callable[[], Awaitable[None]],
    *,
    after_seconds: float,
) -> T:
    """Run `work`, saying the holding line once if it is still running after `after_seconds`."""

    async def hold() -> None:
        await anyio.sleep(after_seconds)
        # Shielded: once it has begun, the work finishing must not cut it off.
        with anyio.CancelScope(shield=True):
            try:
                await say()
            except Exception:  # noqa: BLE001 - a courtesy must not fail the turn
                logger.warning("The holding line could not be spoken.", exc_info=True)

    failure: Exception | None = None
    result: T | None = None
    async with anyio.create_task_group() as group:
        group.start_soon(hold)
        try:
            result = await work()
        except Exception as exc:  # noqa: BLE001 - re-raised below, unwrapped
            # Task groups wrap an exception from their body in an ExceptionGroup; the caller
            # asked to await `work`, so it gets `work`'s own exception.
            failure = exc
        finally:
            # Cancels the sleep if the work finished first; a line already
            # being spoken is shielded and finishes before the group exits.
            group.cancel_scope.cancel()
    if failure is not None:
        raise failure
    return result  # type: ignore[return-value]
