"""Socket.IO transport for the agentix runtime — msgpack payloads.

Every event's payload is a single `bytes` arg = msgpack-packed dict.
Socket.IO's binary support handles the wire framing automatically;
clients send bytes, server receives bytes, vice versa.

Wire (Socket.IO event names + payload dicts):

  client → server:
    "stream"           {call_id, package, method, args, kwargs}
    "bidi:start"       {call_id, package, method, args, kwargs}
    "bidi:in"          {call_id, item}
    "bidi:end_in"      {call_id}
    "cancel"           {call_id}
    "logs:subscribe"   {filter?: str}   # logger-name prefix (default: "agentix")
    "logs:unsubscribe" {}
    "traces:subscribe" {}
    "traces:unsubscribe" {}

  server → client:
    STREAM_ITEM     {call_id, value}
    STREAM_END      {call_id}
    STREAM_ERROR    {call_id, error}
    BIDI_OUT        {call_id, value}
    BIDI_END        {call_id}
    BIDI_ERROR      {call_id, error}
    LOG             {level, name, message, timestamp}
    TRACE           {kind, payload, call_id, source, timestamp}
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import socketio
from pydantic import ValidationError

from agentix.idents import CallId
from agentix.runtime import _pump
from agentix.runtime.codec import pack, unpack
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


def _u(data: Any) -> dict:
    """Unpack a Socket.IO event payload (msgpack bytes). Missing → {}."""
    if not data:
        return {}
    return unpack(bytes(data)) or {}


@dataclass
class _CallState:
    """Per-(session, call_id) state for an in-flight stream/bidi call.

    Bidi inbound path is two-tier (mirroring `agentix.runtime.worker`'s
    pump pattern): `intake` is unbounded so `on_bidi_in` is a sync
    `put_nowait` — no await, no reordering even under `async_handlers=True`.
    A per-call `pump` task drains intake into the bounded `in_queue`
    that the dispatcher's `_input_iter` reads from; the pump's
    `await put` is what gives backpressure through the wire."""

    task: asyncio.Task
    intake: asyncio.Queue | None = None
    in_queue: asyncio.Queue | None = None
    pump: asyncio.Task | None = None
    in_sentinel: Any = None
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

    # `async_handlers=True` (the default) spawns one task per event, so
    # CANCEL can interrupt an in-flight bidi flow even if BIDI_IN
    # handlers are queued up. We keep BIDI_IN ordering by making
    # `on_bidi_in` synchronous (`put_nowait` to an unbounded intake) and
    # moving the only blocking `await put` onto a dedicated per-call
    # pump task. See `_CallState` for the two-tier queue design.
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
    sessions: dict[str, _SessionState] = {}

    def _event(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return sio.on(name)  # type: ignore[reportReturnType]

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
            _pump.cancel_if_running(call.pump)
        await _drain_tasks(
            [c.task for c in sess.calls.values()]
            + [c.pump for c in sess.calls.values() if c.pump is not None],
        )
        if sess.log_handler is not None:
            logging.getLogger(_ROOT_LOG_NAME).removeHandler(sess.log_handler)
        logger.debug("sio disconnect %s", sid)

    # ── server-streaming ─────────────────────────────────────────

    @_event(STREAM)
    async def on_stream(sid: str, data: Any) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        payload = _u(data)
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            await sio.emit(STREAM_ERROR, pack({
                "call_id": "", "error": {"type": "BadRequest", "message": "missing call_id"},
            }), to=sid)
            return

        async def _drive() -> None:
            try:
                request = RemoteRequest(
                    package=payload["package"], method=payload["method"],
                    args=payload.get("args") or [],
                    kwargs=payload.get("kwargs") or {},
                    call_id=CallId(payload.get("trace_call_id") or call_id),
                )
            except (KeyError, ValidationError) as exc:
                await sio.emit(STREAM_ERROR, pack({
                    "call_id": call_id,
                    "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
                }), to=sid)
                return
            async for ev in multiplexer.dispatch_stream(request):
                kind = ev.get("type")
                if kind == "item":
                    await sio.emit(STREAM_ITEM, pack({"call_id": call_id, "value": ev.get("value")}), to=sid)
                elif kind == "end":
                    await sio.emit(STREAM_END, pack({"call_id": call_id}), to=sid)
                elif kind == "error":
                    await sio.emit(STREAM_ERROR, pack({"call_id": call_id, "error": ev.get("error")}), to=sid)

        await _spawn_call(sess, call_id, _drive())

    # ── bidi ─────────────────────────────────────────────────────

    @_event(BIDI_START)
    async def on_bidi_start(sid: str, data: Any) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        payload = _u(data)
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            await sio.emit(BIDI_ERROR, pack({
                "call_id": "", "error": {"type": "BadRequest", "message": "missing call_id"},
            }), to=sid)
            return

        try:
            request = RemoteRequest(
                package=payload["package"], method=payload["method"],
                args=payload.get("args") or [],
                kwargs=payload.get("kwargs") or {},
                call_id=CallId(payload.get("trace_call_id") or call_id),
            )
        except (KeyError, ValidationError) as exc:
            await sio.emit(BIDI_ERROR, pack({
                "call_id": call_id,
                "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
            }), to=sid)
            return

        intake: asyncio.Queue = asyncio.Queue()
        in_queue: asyncio.Queue = asyncio.Queue(maxsize=_pump.DEFAULT_BIDI_BUFFER)
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
                    await sio.emit(BIDI_OUT, pack({"call_id": call_id, "value": ev.get("value")}), to=sid)
                elif kind == "end":
                    await sio.emit(BIDI_END, pack({"call_id": call_id}), to=sid)
                elif kind == "error":
                    await sio.emit(BIDI_ERROR, pack({"call_id": call_id, "error": ev.get("error")}), to=sid)

        pump_task = asyncio.create_task(_pump.drain(intake, in_queue, sentinel))
        call_state = _CallState(
            task=None,                  # type: ignore[arg-type]  filled in by _spawn_call
            intake=intake,
            in_queue=in_queue,
            pump=pump_task,
            in_sentinel=sentinel,
        )
        await _spawn_call(sess, call_id, _drive(), state=call_state)

    @_event(BIDI_IN)
    async def on_bidi_in(sid: str, data: Any) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        payload = _u(data)
        call_id = payload.get("call_id")
        call = sess.calls.get(call_id) if isinstance(call_id, str) else None
        if call is None or call.intake is None or call.in_done:
            return
        # Synchronous put_nowait on unbounded intake — no await, so the
        # handler completes atomically. Order is preserved across
        # concurrent handler tasks under `async_handlers=True`.
        call.intake.put_nowait(payload.get("item"))

    @_event(BIDI_END_IN)
    async def on_bidi_end_in(sid: str, data: Any) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        payload = _u(data)
        call_id = payload.get("call_id")
        call = sess.calls.get(call_id) if isinstance(call_id, str) else None
        if call is None or call.intake is None or call.in_done:
            return
        if call.in_sentinel is not None:
            call.intake.put_nowait(call.in_sentinel)
            call.in_done = True

    # ── cancel ───────────────────────────────────────────────────

    @_event(CANCEL)
    async def on_cancel(sid: str, data: Any) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        payload = _u(data)
        call_id = payload.get("call_id")
        call = sess.calls.pop(call_id, None) if isinstance(call_id, str) else None
        if call is not None:
            call.task.cancel()
            _pump.cancel_if_running(call.pump)

    # ── logs ─────────────────────────────────────────────────────

    @_event(LOGS_SUBSCRIBE)
    async def on_logs_subscribe(sid: str, data: Any = None) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        if sess.log_handler is not None:
            return  # idempotent
        prefix = _u(data).get("filter") or _ROOT_LOG_NAME
        handler = _make_log_forwarder(sio, sid, prefix)
        logging.getLogger(_ROOT_LOG_NAME).addHandler(handler)
        sess.log_handler = handler

    @_event(LOGS_UNSUBSCRIBE)
    async def on_logs_unsubscribe(sid: str, _data: Any = None) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        if sess.log_handler is not None:
            logging.getLogger(_ROOT_LOG_NAME).removeHandler(sess.log_handler)
            sess.log_handler = None

    # ── traces ──────────────────────────────────────────────────

    @_event(TRACES_SUBSCRIBE)
    async def on_traces_subscribe(sid: str, _data: Any = None) -> None:
        await sio.enter_room(sid, TRACES_ROOM)

    @_event(TRACES_UNSUBSCRIBE)
    async def on_traces_unsubscribe(sid: str, _data: Any = None) -> None:
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
            popped = sess.calls.pop(call_id, None)
            # Pump task outlives the dispatch task only until the next item
            # arrives at intake. Cancel it explicitly so it doesn't sit
            # idle waiting on an intake that will never get more frames.
            if popped is not None:
                _pump.cancel_if_running(popped.pump)

        task.add_done_callback(_on_done)

    asgi_app = socketio.ASGIApp(sio, socketio_path="/socket.io")
    return sio, asgi_app


# ── log forwarding ───────────────────────────────────────────────


def _make_log_forwarder(sio: socketio.AsyncServer, sid: str, prefix: str) -> logging.Handler:
    """A logging.Handler that schedules `sio.emit("log", msgpack(...))` for each record
    whose logger name starts with `prefix`."""

    class _Forwarder(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if not record.name.startswith(prefix):
                return
            try:
                msg = self.format(record)
            except Exception:
                msg = record.getMessage()
            payload = pack({
                "level": record.levelname,
                "name": record.name,
                "message": msg,
                "timestamp": record.created,
            })
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
