"""Bounding how long a streaming provider may say nothing.

A socket that opens and then goes quiet is the failure mode neither vendor
protocol has an answer for: there is no error, no close frame and nothing to
catch — the turn simply never finishes, with a caller listening to silence.

`messages` puts a clock between messages rather than around the whole stream.
A synthesiser sending audio steadily is working however long the reply is;
one that has sent nothing for the idle timeout has stopped, whatever it
intends to do later. The `TimeoutError` it raises is caught by each adapter's
existing handler and becomes that adapter's own "unavailable" error, so a
provider that goes quiet fails exactly like one that disconnects.

Nothing here reconnects. That was a milestone-7 decision and it stands.
"""

from collections.abc import AsyncIterator
from typing import Any

import anyio


async def messages(connection: Any, idle_timeout: float) -> AsyncIterator[Any]:
    """Every message, so long as none of them takes too long to arrive."""
    iterator = connection.__aiter__()
    while True:
        with anyio.fail_after(idle_timeout):
            try:
                message = await iterator.__anext__()
            except StopAsyncIteration:
                return
        yield message


__all__ = ["messages"]
