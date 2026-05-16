"""Socket.IO transport for the agentix runtime.

Single Socket.IO connection per `RuntimeClient` multiplexes:

  - server-streaming calls (impl returns `AsyncIterator[T]`)
  - bidirectional calls (impl takes `AsyncIterator[T]` and returns `AsyncIterator[U]`)
  - log subscription (forwards `logging.Handler` records to the client)

The Socket.IO ASGI app is mounted alongside FastAPI in `app.py`; HTTP
unary RPC stays on `POST /_remote`. Inputs and outputs are correlated by
a caller-generated `call_id` so multiple concurrent calls coexist on one
connection.

Internally, every streaming / bidi event is just forwarded between the
SIO transport and the per-namespace worker via the `NamespaceMultiplexer`.
The multiplexer's dispatch_stream / dispatch_bidi yield event dicts
that this transport translates to Socket.IO emits.

Wire (Socket.IO event names + payloads — all JSON):

  client → server:
    "stream"           {call_id, package, method, args, kwargs}
    "bidi:start"       {call_id, package, method, args, kwargs}
    "bidi:in"          {call_id, item}
    "bidi:end_in"      {call_id}
    "cancel"           {call_id}
    "logs:subscribe"   {filter?: str}   # logger-name prefix (default: "agentix")
    "logs:unsubscribe" {}

  server → client:
    STREAM_ITEM     {call_id, value}
    STREAM_END      {call_id}
    STREAM_ERROR    {call_id, error: RemoteError}
    BIDI_OUT        {call_id, value}
    BIDI_END        {call_id}
    BIDI_ERROR      {call_id, error: RemoteError}
    "log"           {level, name, message, timestamp}
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

import socketio
from pydantic import ValidationError

from agentix.runtime.events import (
    BIDI_END,
    BIDI_END_IN,
    BIDI_ERROR,
    BIDI_IN,
    BIDI_OUT,
    BIDI_START,
    CANCEL,
    LOG,
    LOGS_SUBSCRIBE,
    LOGS_UNSUBSCRIBE,
    STREAM,
    STREAM_END,
    STREAM_ERROR,
    STREAM_ITEM,
    TRACES_ROOM,
    TRACES_SUBSCRIBE,
    TRACES_UNSUBSCRIBE,
)
from agentix.runtime.models import RemoteError, RemoteRequest
from agentix.runtime.multiplexer import NamespaceMultiplexer

logger = logging.getLogger("agentix.runtime.sio")

_ROOT_LOG_NAME = "agentix"  # forwarded log records start at this logger and below


@dataclass
class _CallState:
    """Per-(session, call_id) state for an in-flight stream/bidi call."""

    task: asyncio.Task
    in_queue: asyncio.Queue | None = None  # only populated for bidi
    in_done: bool = False


@dataclass
class _SessionState:
    """Per-Socket.IO-session bookkeeping."""

    calls: dict[str, _CallState] = field(default_factory=dict)
    log_handler: logging.Handler | None = None


def make_sio(
    multiplexer: NamespaceMultiplexer,
) -> tuple[socketio.AsyncServer, socketio.ASGIApp]:
    """Build the Socket.IO AsyncServer wired to `multiplexer`, plus its ASGI app."""

    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    sessions: dict[str, _SessionState] = {}

    # ── connection lifecycle ─────────────────────────────────────

    @sio.event
    async def connect(sid: str, environ: dict, auth: Any = None) -> None:
        sessions[sid] = _SessionState()
        logger.debug("sio connect %s", sid)

    @sio.event
    async def disconnect(sid: str) -> None:
        sess = sessions.pop(sid, None)
        if sess is None:
            return
        for call in sess.calls.values():
            call.task.cancel()
        await _drain_tasks([c.task for c in sess.calls.values()])
        if sess.log_handler is not None:
            logging.getLogger(_ROOT_LOG_NAME).removeHandler(sess.log_handler)
        logger.debug("sio disconnect %s", sid)

    # ── server-streaming ─────────────────────────────────────────

    @sio.on(STREAM)
    async def on_stream(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        if not isinstance(call_id, str):
            await sio.emit(STREAM_ERROR, {
                "call_id": "", "error": {"type": "BadRequest", "message": "missing call_id"},
            }, to=sid)
            return

        async def _drive() -> None:
            try:
                request = RemoteRequest(
                    package=data["package"], method=data["method"],
                    args=data.get("args", []) or [],
                    kwargs=data.get("kwargs", {}) or {},
                    call_id=call_id,
                )
            except (KeyError, ValidationError) as exc:
                await sio.emit(STREAM_ERROR, {
                    "call_id": call_id,
                    "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
                }, to=sid)
                return
            async for ev in multiplexer.dispatch_stream(request):
                kind = ev.get("type")
                if kind == "item":
                    await sio.emit(STREAM_ITEM, {"call_id": call_id, "value": ev.get("value")}, to=sid)
                elif kind == "end":
                    await sio.emit(STREAM_END, {"call_id": call_id}, to=sid)
                elif kind == "error":
                    await sio.emit(STREAM_ERROR, {"call_id": call_id, "error": ev.get("error")}, to=sid)

        await _spawn_call(sess, call_id, _drive())

    # ── bidi ─────────────────────────────────────────────────────

    @sio.on(BIDI_START)
    async def on_bidi_start(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        if not isinstance(call_id, str):
            await sio.emit(BIDI_ERROR, {
                "call_id": "", "error": {"type": "BadRequest", "message": "missing call_id"},
            }, to=sid)
            return

        try:
            request = RemoteRequest(
                package=data["package"], method=data["method"],
                args=data.get("args", []) or [],
                kwargs=data.get("kwargs", {}) or {},
                call_id=call_id,
            )
        except (KeyError, ValidationError) as exc:
            await sio.emit(BIDI_ERROR, {
                "call_id": call_id,
                "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
            }, to=sid)
            return

        in_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        sentinel = object()

        async def _input_iter():
            while True:
                item = await in_queue.get()
                if item is sentinel:
                    return
                yield item

        async def _drive() -> None:
            async for ev in multiplexer.dispatch_bidi(request, _input_iter()):
                kind = ev.get("type")
                if kind == "item":
                    await sio.emit(BIDI_OUT, {"call_id": call_id, "value": ev.get("value")}, to=sid)
                elif kind == "end":
                    await sio.emit(BIDI_END, {"call_id": call_id}, to=sid)
                elif kind == "error":
                    await sio.emit(BIDI_ERROR, {"call_id": call_id, "error": ev.get("error")}, to=sid)

        # Pre-create the state so input handlers can push to the queue
        # even if they arrive before _drive starts iterating.
        call_state = _CallState(task=None, in_queue=in_queue)  # type: ignore[arg-type]
        call_state.in_sentinel = sentinel  # type: ignore[attr-defined]
        await _spawn_call(sess, call_id, _drive(), state=call_state)

    @sio.on(BIDI_IN)
    async def on_bidi_in(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        call = sess.calls.get(call_id) if isinstance(call_id, str) else None
        if call is None or call.in_queue is None or call.in_done:
            return
        # Worker validates input; runtime just forwards the raw item.
        await call.in_queue.put(data.get("item"))

    @sio.on(BIDI_END_IN)
    async def on_bidi_end_in(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        call = sess.calls.get(call_id) if isinstance(call_id, str) else None
        if call is None or call.in_queue is None or call.in_done:
            return
        sentinel = getattr(call, "in_sentinel", None)
        if sentinel is not None:
            await call.in_queue.put(sentinel)
            call.in_done = True

    # ── cancel ───────────────────────────────────────────────────

    @sio.on(CANCEL)
    async def on_cancel(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        call = sess.calls.pop(call_id, None) if isinstance(call_id, str) else None
        if call is not None:
            call.task.cancel()

    # ── logs ─────────────────────────────────────────────────────

    @sio.on(LOGS_SUBSCRIBE)
    async def on_logs_subscribe(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        if sess.log_handler is not None:
            return  # idempotent
        prefix = data.get("filter") or _ROOT_LOG_NAME
        handler = _make_log_forwarder(sio, sid, prefix)
        logging.getLogger(_ROOT_LOG_NAME).addHandler(handler)
        sess.log_handler = handler

    @sio.on(LOGS_UNSUBSCRIBE)
    async def on_logs_unsubscribe(sid: str, _data: dict[str, Any] | None = None) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        if sess.log_handler is not None:
            logging.getLogger(_ROOT_LOG_NAME).removeHandler(sess.log_handler)
            sess.log_handler = None

    # ── traces ──────────────────────────────────────────────────

    @sio.on(TRACES_SUBSCRIBE)
    async def on_traces_subscribe(sid: str, _data: dict[str, Any] | None = None) -> None:
        await sio.enter_room(sid, TRACES_ROOM)

    @sio.on(TRACES_UNSUBSCRIBE)
    async def on_traces_unsubscribe(sid: str, _data: dict[str, Any] | None = None) -> None:
        await sio.leave_room(sid, TRACES_ROOM)

    # ── helpers (closure over sio + sessions) ────────────────────

    async def _spawn_call(
        sess: _SessionState, call_id: str, coro, *, state: _CallState | None = None,
    ) -> None:
        task = asyncio.create_task(coro)
        if state is None:
            state = _CallState(task=task)
        else:
            state.task = task
        sess.calls[call_id] = state

        def _on_done(_t: asyncio.Task) -> None:
            sess.calls.pop(call_id, None)

        task.add_done_callback(_on_done)

    asgi_app = socketio.ASGIApp(sio, socketio_path="/socket.io")
    return sio, asgi_app


# ── log forwarding ───────────────────────────────────────────────


def _make_log_forwarder(sio: socketio.AsyncServer, sid: str, prefix: str) -> logging.Handler:
    """A logging.Handler that schedules `sio.emit("log", ...)` for each record
    whose logger name starts with `prefix`."""

    class _Forwarder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if not record.name.startswith(prefix):
                return
            try:
                msg = self.format(record)
            except Exception:
                msg = record.getMessage()
            payload = {
                "level": record.levelname,
                "name": record.name,
                "message": msg,
                "timestamp": record.created,
            }
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no loop on this thread → drop
            loop.create_task(sio.emit(LOG, payload, to=sid))

    h = _Forwarder()
    h.setLevel(logging.DEBUG)
    return h


async def _drain_tasks(tasks: list[asyncio.Task]) -> None:
    """Best-effort cancel + await for a batch of tasks."""
    if not tasks:
        return
    for t in tasks:
        if not t.done():
            t.cancel()
    with contextlib.suppress(BaseException):
        await asyncio.gather(*tasks, return_exceptions=True)
