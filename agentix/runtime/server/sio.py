"""Socket.IO transport for the agentix runtime.

Single Socket.IO connection per `RuntimeClient` multiplexes:

  - server-streaming calls (impl returns `AsyncIterator[T]`)
  - bidirectional calls (impl takes `AsyncIterator[T]` and returns `AsyncIterator[U]`)
  - log subscription (forwards `logging.Handler` records to the client)

The Socket.IO ASGI app is mounted alongside FastAPI in `server.py`; HTTP
unary RPC stays on `POST /_remote`. Inputs and outputs are correlated by
a caller-generated `call_id` so multiple concurrent calls coexist on one
connection.

Wire (Socket.IO event names + payloads — all JSON):

  client → server:
    "stream"          {call_id, package, method, args, kwargs}
    "bidi:start"      {call_id, package, method, args, kwargs}
    "bidi:in"         {call_id, item}
    "bidi:end_in"     {call_id}
    "cancel"          {call_id}
    "logs:subscribe"  {filter?: str}   # logger-name prefix filter (default: "agentix")
    "logs:unsubscribe" {}

  server → client:
    "stream:item"     {call_id, value}
    "stream:end"      {call_id}
    "stream:error"    {call_id, error: RemoteError}
    "bidi:out"        {call_id, value}
    "bidi:end"        {call_id}
    "bidi:error"      {call_id, error: RemoteError}
    "log"             {level, name, message, timestamp}
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

import socketio
from pydantic import ValidationError

from agentix.dispatch import Registry
from agentix.runtime.models import RemoteError, RemoteRequest

logger = logging.getLogger("agentix.runtime.sio")


_ROOT_LOG_NAME = "agentix"  # forwarded log records start at this logger and below


@dataclass
class _CallState:
    """Per-(session, call_id) state for an in-flight stream/bidi call."""

    task: asyncio.Task
    in_queue: asyncio.Queue | None = None  # only populated for bidi


@dataclass
class _SessionState:
    """Per-Socket.IO-session bookkeeping."""

    calls: dict[str, _CallState] = field(default_factory=dict)
    log_handler: logging.Handler | None = None


def make_sio(registry: Registry) -> tuple[socketio.AsyncServer, socketio.ASGIApp]:
    """Build the Socket.IO AsyncServer wired to `registry`, plus its ASGI app.

    Returns (sio_server, asgi_app). The asgi_app is meant to be wrapped
    around the FastAPI app: `socketio.ASGIApp(sio, fastapi_app)`.
    """
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

    @sio.on("stream")
    async def on_stream(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        if not isinstance(call_id, str):
            await sio.emit("stream:error", {
                "call_id": "", "error": {"type": "BadRequest", "message": "missing call_id"},
            }, to=sid)
            return

        async def _drive() -> None:
            try:
                request = RemoteRequest(
                    package=data["package"], method=data["method"],
                    args=data.get("args", []) or [],
                    kwargs=data.get("kwargs", {}) or {},
                )
            except (KeyError, ValidationError) as exc:
                await sio.emit("stream:error", {
                    "call_id": call_id,
                    "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
                }, to=sid)
                return
            try:
                dispatcher = await registry.get_or_load(request.package)
            except Exception as exc:
                await sio.emit("stream:error", {
                    "call_id": call_id,
                    "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
                }, to=sid)
                return
            if dispatcher is None:
                await sio.emit("stream:error", {
                    "call_id": call_id,
                    "error": RemoteError(type="PackageNotLoaded",
                                         message=f"closure not loaded: {request.package!r}").model_dump(),
                }, to=sid)
                return
            if dispatcher.is_bidi(request.method):
                await sio.emit("stream:error", {
                    "call_id": call_id,
                    "error": RemoteError(type="MethodIsBidi",
                                         message=f"{request.method} is bidirectional; use bidi:start").model_dump(),
                }, to=sid)
                return
            async for event in dispatcher.dispatch_stream(request):
                if "item" in event:
                    await sio.emit("stream:item", {"call_id": call_id, "value": event["item"]}, to=sid)
                elif "end" in event:
                    await sio.emit("stream:end", {"call_id": call_id}, to=sid)
                elif "error" in event:
                    await sio.emit("stream:error", {"call_id": call_id, "error": event["error"]}, to=sid)

        await _spawn_call(sess, call_id, _drive())

    # ── bidi ─────────────────────────────────────────────────────

    @sio.on("bidi:start")
    async def on_bidi_start(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        if not isinstance(call_id, str):
            await sio.emit("bidi:error", {
                "call_id": "", "error": {"type": "BadRequest", "message": "missing call_id"},
            }, to=sid)
            return

        try:
            request = RemoteRequest(
                package=data["package"], method=data["method"],
                args=data.get("args", []) or [],
                kwargs=data.get("kwargs", {}) or {},
            )
        except (KeyError, ValidationError) as exc:
            await sio.emit("bidi:error", {
                "call_id": call_id,
                "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
            }, to=sid)
            return

        in_queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        async def _drive() -> None:
            sentinel = object()

            async def _input_iter():
                while True:
                    item = await in_queue.get()
                    if item is sentinel:
                        return
                    yield item

            try:
                dispatcher = await registry.get_or_load(request.package)
            except Exception as exc:
                await sio.emit("bidi:error", {
                    "call_id": call_id,
                    "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
                }, to=sid)
                return
            if dispatcher is None:
                await sio.emit("bidi:error", {
                    "call_id": call_id,
                    "error": RemoteError(type="PackageNotLoaded",
                                         message=f"closure not loaded: {request.package!r}").model_dump(),
                }, to=sid)
                return
            if not dispatcher.is_bidi(request.method):
                await sio.emit("bidi:error", {
                    "call_id": call_id,
                    "error": RemoteError(type="NotABidiMethod",
                                         message=f"{request.method} is not bidirectional").model_dump(),
                }, to=sid)
                return

            input_adapter = dispatcher.input_adapter_for(request.method)
            # Sentinel value on queue closes the input iterator from the on_bidi_end_in
            # event or from disconnect cleanup. Store it on the state so handlers can
            # push it.
            call_state.in_queue = in_queue  # noqa: F841 — closure-captures call_state below
            call_state.in_sentinel = sentinel  # type: ignore[attr-defined]
            call_state.input_adapter = input_adapter  # type: ignore[attr-defined]

            async for event in dispatcher.dispatch_bidi(request, _input_iter()):
                if "item" in event:
                    await sio.emit("bidi:out", {"call_id": call_id, "value": event["item"]}, to=sid)
                elif "end" in event:
                    await sio.emit("bidi:end", {"call_id": call_id}, to=sid)
                elif "error" in event:
                    await sio.emit("bidi:error", {"call_id": call_id, "error": event["error"]}, to=sid)

        # Pre-create the state so input handlers can push to the queue even if
        # they arrive before _drive starts iterating.
        call_state = _CallState(task=None, in_queue=in_queue)  # type: ignore[arg-type]
        await _spawn_call(sess, call_id, _drive(), state=call_state)

    @sio.on("bidi:in")
    async def on_bidi_in(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        call = sess.calls.get(call_id) if isinstance(call_id, str) else None
        if call is None or call.in_queue is None:
            return
        raw = data.get("item")
        adapter = getattr(call, "input_adapter", None)
        try:
            item = adapter.validate_python(raw) if adapter is not None else raw
        except ValidationError as exc:
            await sio.emit("bidi:error", {
                "call_id": call_id,
                "error": RemoteError(type="InputValidation", message=str(exc)).model_dump(),
            }, to=sid)
            sentinel = getattr(call, "in_sentinel", None)
            if sentinel is not None:
                await call.in_queue.put(sentinel)
            return
        await call.in_queue.put(item)

    @sio.on("bidi:end_in")
    async def on_bidi_end_in(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        call = sess.calls.get(call_id) if isinstance(call_id, str) else None
        if call is None or call.in_queue is None:
            return
        sentinel = getattr(call, "in_sentinel", None)
        if sentinel is not None:
            await call.in_queue.put(sentinel)

    # ── cancel ───────────────────────────────────────────────────

    @sio.on("cancel")
    async def on_cancel(sid: str, data: dict[str, Any]) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        call_id = data.get("call_id")
        call = sess.calls.pop(call_id, None) if isinstance(call_id, str) else None
        if call is not None:
            call.task.cancel()

    # ── logs ─────────────────────────────────────────────────────

    @sio.on("logs:subscribe")
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

    @sio.on("logs:unsubscribe")
    async def on_logs_unsubscribe(sid: str, _data: dict[str, Any] | None = None) -> None:
        sess = sessions.get(sid)
        if sess is None:
            return
        if sess.log_handler is not None:
            logging.getLogger(_ROOT_LOG_NAME).removeHandler(sess.log_handler)
            sess.log_handler = None

    # ── traces ──────────────────────────────────────────────────
    #
    # Trace events come from closures via `agentix.trace.emit(...)`. The
    # runtime broadcasts each event to clients in the "traces" room.

    @sio.on("traces:subscribe")
    async def on_traces_subscribe(sid: str, _data: dict[str, Any] | None = None) -> None:
        await sio.enter_room(sid, "traces")

    @sio.on("traces:unsubscribe")
    async def on_traces_unsubscribe(sid: str, _data: dict[str, Any] | None = None) -> None:
        await sio.leave_room(sid, "traces")

    # ── helpers (closure over sio + sessions) ────────────────────

    async def _spawn_call(sess: _SessionState, call_id: str, coro, *, state: _CallState | None = None) -> None:
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
    whose logger name starts with `prefix`. Option B per the design: any log
    going through the `agentix` (or `prefix`-rooted) logger tree is forwarded.

    Closure stdout/stderr via `print()` is NOT captured by this handler —
    closures should use `logging.getLogger(__name__).info(...)`. A separate
    stdout-capture handler can be added later if needed.
    """

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
            # `emit` must be schedulable without awaiting — logging is sync.
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                return
            if loop.is_running():
                loop.create_task(sio.emit("log", payload, to=sid))

    h = _Forwarder()
    h.setLevel(logging.DEBUG)  # forward everything; client can filter
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
