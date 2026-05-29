"""Host-side namespace helpers.

`AsyncClientNamespace` is a thin subclass of `socketio.AsyncClientNamespace`
that msgpack-wraps event payloads — so plugin authors write
`await self.emit("x", {"a": 1})` and `async def on_x(self, data)`, and
the bytes/msgpack wire format stays internal.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import socketio

from agentix.runtime.shared.codec import pack, unpack

logger = logging.getLogger("agentix.runtime.client.sio")


def _decode(raw: Any) -> Any:
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes):
        return unpack(raw)
    return raw


# Socket.IO lifecycle events run inline (they're cheap and ordering
# matters); everything else is a user data event and is detached.
_LIFECYCLE_EVENTS = frozenset({"connect", "disconnect", "connect_error"})


class AsyncClientNamespace(socketio.AsyncClientNamespace):
    """`socketio.AsyncClientNamespace` with msgpack at the boundary.

    Override `on_<event>` for inbound; call `await self.emit(...)` for
    outbound. Data is plain Python — packing happens automatically.

    Data-event handlers are dispatched as **detached tasks**, never
    awaited inline. `socketio.AsyncClient` awaits `trigger_event` inside
    its single websocket receive loop — so a slow handler (e.g. one
    that calls a slow LLM) would stall *every* inbound event on the
    connection, including unrelated `c.remote` results. Detaching keeps
    the receive loop free; handler ordering per event is still
    preserved by the order tasks are created.
    """

    _detached_tasks: set[asyncio.Task]

    async def emit(self, event: str, data: Any = None, **kwargs: Any) -> Any:
        return await super().emit(event, pack(data), **kwargs)

    async def trigger_event(self, event: str, *args: Any) -> Any:
        if event in _LIFECYCLE_EVENTS:
            # Lifecycle: run inline. No msgpack payload to unwrap.
            return await super().trigger_event(event, *args)

        # Data event: unwrap the msgpack payload, then dispatch detached.
        if args and isinstance(args[0], (bytes, bytearray, memoryview)):
            args = (_decode(args[0]),) + args[1:]

        handler = getattr(self, "on_" + event, None)
        if handler is None:
            return None

        result = handler(*args)
        if asyncio.iscoroutine(result):
            if not hasattr(self, "_detached_tasks"):
                self._detached_tasks = set()
            task = asyncio.create_task(self._guard(result, event))
            self._detached_tasks.add(task)
            task.add_done_callback(self._detached_tasks.discard)
        return None

    @staticmethod
    async def _guard(coro: Any, event: str) -> None:
        try:
            await coro
        except Exception:
            logger.exception("namespace handler for %r raised", event)


RequestMethod = Callable[[Any, Any], Awaitable[Any]]
WrappedRequestHandler = Callable[[Any, Any], Awaitable[None]]


def request_handler(event: str) -> Callable[[RequestMethod], WrappedRequestHandler]:
    """Decorate a host `on_<event>` coroutine that answers a sandbox
    `Namespace.request(event, body)` round-trip.

    The wrapped coroutine receives the unwrapped request *body* and returns
    the reply value. The request envelope (`request_id` / `data`), the reply
    event name (`<event>:result` / `<event>:error`), and error replies are
    all handled automatically — so a typo in the reply event name or a
    forgotten reply (which otherwise hangs the sandbox until its request
    timeout) can't happen. A raised exception becomes an `<event>:error`
    reply carrying `{"type", "message"}` (the shape `Namespace.request`
    raises as `RemoteSioError`).

        class MyHost(AsyncClientNamespace):
            def __init__(self) -> None:
                super().__init__("/my-plugin")

            @request_handler("fetch")
            async def on_fetch(self, body):
                return await do_work(body)
    """

    def decorate(method: RequestMethod) -> WrappedRequestHandler:
        @wraps(method)
        async def wrapper(self: Any, payload: Any) -> None:
            request_id = payload.get("request_id") if isinstance(payload, dict) else None
            body = payload.get("data") if isinstance(payload, dict) else payload
            try:
                value = await method(self, body)
            except Exception as exc:
                await self.emit(
                    f"{event}:error",
                    {"request_id": request_id, "error": {"type": type(exc).__name__, "message": str(exc)}},
                )
                return
            await self.emit(f"{event}:result", {"request_id": request_id, "value": value})

        return wrapper

    return decorate


__all__ = ["AsyncClientNamespace", "request_handler"]
