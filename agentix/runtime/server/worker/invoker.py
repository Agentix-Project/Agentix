"""Worker-side callable execution.

The worker unpickles the callable + args/kwargs, calls it, and pickles
the result back. No shape detection, no TypeAdapter validation — pickle
preserves Python object identity end to end.

A coroutine function is awaited directly on the worker's event loop; a
plain (sync) function runs in a thread via ``asyncio.to_thread`` so a
blocking body can never stall the loop — and therefore can never stall
concurrent calls or the ``/log`` / ``/trace`` side channels. The
per-call ``DISPATCH_CALL_ID`` contextvar is copied into the thread by
``to_thread``, so log/trace stamping still works.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import pickle
import traceback
from typing import Any

from agentix.runtime.shared.callables import display_name_for
from agentix.runtime.shared.models import RemoteError, RemoteRequest, RemoteResponse

logger = logging.getLogger("agentix.runtime.server.worker.invoker")


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
