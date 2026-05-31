"""Worker-side callable execution.

The worker unpickles the callable + args/kwargs, calls it, and pickles
the result back. No shape detection, no TypeAdapter validation — pickle
preserves Python object identity end to end.

A coroutine function is awaited directly on the worker's event loop; a
plain (sync) function runs in a thread via ``asyncio.to_thread`` so a
blocking body can never stall the loop — and therefore can never stall
concurrent calls or the ``/log`` / ``/trace`` side channels.

Before ``fn`` runs the invoker establishes the per-call *dispatch
scope*: it stamps the ``DISPATCH_CALL_ID`` contextvar (correlates
worker-emitted spans/logs to the originating ``c.remote(...)``) and
``attach``es the host's ambient ``agentix.utils.context`` carrier
(baggage + propagator slices, e.g. the active trace scope, so spans
opened in ``fn`` nest under the host's span). ``asyncio.to_thread``
copies the current contextvars into the thread, so the scope holds for
sync bodies too.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import pickle
import traceback
from collections.abc import Iterator
from typing import Any

from agentix.runtime.shared.callables import display_name_for
from agentix.runtime.shared.models import RemoteError, RemoteRequest, RemoteResponse
from agentix.utils import context as _context
from agentix.utils.trace._bridge import DISPATCH_CALL_ID

logger = logging.getLogger("agentix.runtime.server.worker.invoker")


@contextlib.contextmanager
def _dispatch_scope(request: RemoteRequest) -> Iterator[None]:
    """Per-call ambient scope: stamp the call id and attach the host's
    propagated context for the duration of the invocation."""
    call_id = str(request.call_id) if request.call_id else None
    tok = DISPATCH_CALL_ID.set(call_id)
    try:
        with _context.attach(_context.decode(request.context)):
            yield
    finally:
        DISPATCH_CALL_ID.reset(tok)


class CallableInvoker:
    """Invoke one resolved Python callable per `RemoteRequest`."""

    async def call(self, fn: Any, request: RemoteRequest) -> RemoteResponse:
        try:
            args, kwargs = pickle.loads(request.arguments)
        except Exception as exc:
            return RemoteResponse(
                ok=False,
                error=RemoteError(
                    type="ArgumentsDecodeError",
                    message=f"failed to unpickle arguments: {exc}",
                ),
            )
        try:
            with _dispatch_scope(request):
                if inspect.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(fn, *args, **kwargs)
                    # A sync callable may still return an awaitable (e.g. a
                    # plain def that returns a coroutine); await it here.
                    if inspect.isawaitable(result):
                        result = await result
        except Exception as exc:
            logger.exception("remote callable '%s' raised", display_name_for(fn))
            return RemoteResponse(
                ok=False,
                error=RemoteError(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                ),
            )
        try:
            payload = pickle.dumps(result)
        except Exception as exc:
            return RemoteResponse(
                ok=False,
                error=RemoteError(
                    type="ResultEncodeError",
                    message=f"failed to pickle return value: {exc}",
                ),
            )
        return RemoteResponse(ok=True, value=payload)


__all__ = ["CallableInvoker"]
