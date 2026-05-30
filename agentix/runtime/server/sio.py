"""Socket.IO transport for the agentix runtime.

Two responsibilities:

1. The RPC protocol on the `/rpc` namespace — `call` / `cancel`
   / `call:result` / `call:error`.

2. Dynamic namespace forwarding. When a worker-side `agentix.Namespace`
   registers via the pipe (`sio_open` frame), this layer registers a
   matching SIO server namespace that forwards inbound events back to
   the worker. Outbound `sio_emit` frames become real SIO emits on the
   corresponding namespace.

Reserved namespace paths (claimed by agentix-core): `/rpc`, `/trace`,
`/log`. Plugins use their own paths (typically `/<package-name>`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import socketio
from pydantic import ValidationError

from agentix.runtime.server.worker import RuntimeWorkerClient, WorkerProcessExited
from agentix.runtime.shared import MAX_MESSAGE_BYTES
from agentix.runtime.shared.callables import RemoteCallable
from agentix.runtime.shared.codec import pack, unpack
from agentix.runtime.shared.idents import CallId
from agentix.runtime.shared.models import RemoteError, RemoteRequest

logger = logging.getLogger("agentix.runtime.sio")
RPC_NAMESPACE = "/rpc"

# Cap on the unacked-result cache. A host that completes calls and never acks
# (a crashed or buggy client) would otherwise pin every result — each holding a
# full pickled return value, up to MAX_MESSAGE_BYTES — in memory forever. Past
# the cap, the oldest unacked entry is evicted.
_MAX_PENDING_RESULTS = 4096


def _u(data: Any) -> dict:
    if not data:
        return {}
    return unpack(bytes(data)) or {}


def _decode(raw: Any) -> Any:
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes):
        return unpack(raw)
    return raw


def _missing_call_id() -> tuple[str, dict[str, Any]]:
    return (
        "call:error",
        {
            "call_id": "",
            "error": {"type": "BadRequest", "message": "missing call_id"},
        },
    )


def _cancelled_error(call_id: str) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "error": RemoteError(
            type="Cancelled",
            message="remote call cancelled",
            cancelled=True,
        ).model_dump(),
    }


def _store_pending_result(
    cache: dict[str, tuple[str, dict[str, Any]]],
    call_id: str,
    value: tuple[str, dict[str, Any]],
    *,
    cap: int = _MAX_PENDING_RESULTS,
) -> str | None:
    """Store a completed result, evicting the oldest entry once `cache` exceeds
    `cap`. Returns the evicted call_id (for the caller to log), else None.

    Relies on dict insertion order: each call_id is stored once, so the first
    key is the oldest unacked result."""
    cache[call_id] = value
    if len(cache) > cap:
        oldest = next(iter(cache))
        cache.pop(oldest, None)
        return oldest
    return None


def make_sio(
    worker: RuntimeWorkerClient,
) -> tuple[socketio.AsyncServer, socketio.ASGIApp]:
    # `namespaces='*'` accepts connects on any namespace path. Plugin
    # namespaces are registered lazily by the worker (`sio_open` frame
    # in response to `agentix.register_namespace(...)`); the host may
    # connect to them before the forwarder is in place. Inbound events
    # are dropped until the forwarder registers, which is what we want.
    #
    # `ping_timeout=300`: a single `c.remote(...)` can run for many
    # minutes (a coding agent, a long eval). During that time the host
    # event loop may briefly be busy enough to delay a pong. A generous
    # timeout keeps the connection alive across those blips — a dropped
    # connection orphans the in-flight call. Genuinely dead peers are
    # still reaped, just after 5 idle minutes instead of 20s.
    #
    # `max_http_buffer_size`: the default 1 MB cap kills the websocket
    # the moment a `c.remote` payload or plugin event exceeds it — see
    # `MAX_MESSAGE_BYTES`.
    sio = socketio.AsyncServer(
        async_mode="asgi",
        cors_allowed_origins="*",
        namespaces="*",
        ping_interval=25,
        ping_timeout=300,
        max_http_buffer_size=MAX_MESSAGE_BYTES,
    )
    # ── execution-once invariant ─────────────────────────────────
    # `_start_call` is the only place a task is created. Every site
    # that may call it (`on_call`, `submit_http_call`) gates on
    # `call_id in calls or call_id in pending_results` first, so a
    # given call_id starts at most one task. Combined with the host
    # generating a fresh call_id per `c.remote(...)`, this guarantees
    # the user-facing contract: each `c.remote(fn, ...)` runs `fn` at
    # most once on the runtime, even across reconnects, replays, and
    # mixed HTTP/SIO submission paths.
    calls: dict[str, asyncio.Task] = {}
    # Completed tasks waiting for the host to ack receipt. The host
    # acks via the `ack` SIO event after consuming the result; only
    # then does the entry leave this dict. This is what makes events
    # survive a connection drop: a reconnecting host emits `resume`
    # and we replay any unacked results for the call_ids it cares
    # about — without ever re-running `fn`.
    pending_results: dict[str, tuple[str, dict[str, Any]]] = {}
    evictions = 0  # count of cap evictions, for throttled warning
    opened_namespaces: set[str] = set()  # paths the worker has opened

    async def _execute_call(payload: dict[str, Any], call_id: str) -> tuple[str, dict[str, Any]]:
        try:
            request = RemoteRequest(
                callable=RemoteCallable(payload["callable"]),
                arguments=payload["arguments"],
                call_id=CallId(call_id),
            )
        except (KeyError, ValidationError) as exc:
            return (
                "call:error",
                {
                    "call_id": call_id,
                    "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
                },
            )

        try:
            resp = await worker.call(request)
        except WorkerProcessExited as exc:
            # Worker died mid-call (crash / OOM-kill / exit). Surface it
            # as a typed error instead of letting the task fail silently
            # (which would hang the host's SIO wait forever). Carry the
            # process exit status so the host can branch on OOM (-9).
            error = RemoteError(
                type="WorkerDied", message=str(exc), returncode=exc.returncode
            ).model_dump()
            return "call:error", {"call_id": call_id, "error": error}
        except Exception as exc:
            error = RemoteError(type=type(exc).__name__, message=str(exc)).model_dump()
            return "call:error", {"call_id": call_id, "error": error}
        if resp.ok:
            return "call:result", {"call_id": call_id, "value": resp.value}
        error = (resp.error or RemoteError(type="Unknown", message="")).model_dump()
        return "call:error", {"call_id": call_id, "error": error}

    async def _emit_task_result(task: asyncio.Task, call_id: str) -> None:
        nonlocal evictions
        if task.cancelled():
            return
        try:
            event, frame = task.result()
        except Exception as exc:
            # Defensive: _execute_call shouldn't raise, but never let a
            # failed task vanish — the host must always get something.
            event = "call:error"
            frame = {
                "call_id": call_id,
                "error": RemoteError(type=type(exc).__name__, message=str(exc)).model_dump(),
            }
        # Store first, emit second. If the host is currently disconnected
        # the emit is a no-op and the cached entry carries the result
        # through to the next `resume`.
        evicted = _store_pending_result(pending_results, call_id, (event, frame))
        if evicted is not None:
            evictions += 1
            # Throttle: warn on the first eviction and periodically after, so
            # sustained overflow doesn't spam a line per completed call.
            if evictions == 1 or evictions % 1000 == 0:
                logger.warning(
                    "pending_results over cap (%d); evicted oldest unacked result(s) "
                    "(total evicted=%d) — a host is completing calls without acking",
                    _MAX_PENDING_RESULTS,
                    evictions,
                )
        with contextlib.suppress(BaseException):
            await sio.emit(event, pack(frame), namespace=RPC_NAMESPACE)

    def _track_call(call_id: str, task: asyncio.Task) -> None:
        calls[call_id] = task
        task.add_done_callback(lambda _t: calls.pop(call_id, None))

    def _start_call(payload: dict[str, Any], call_id: str) -> asyncio.Task:
        task = asyncio.create_task(_execute_call(payload, call_id))
        _track_call(call_id, task)
        return task

    async def submit_http_call(payload: dict[str, Any], *, wait_s: float = 1.0) -> dict[str, Any]:
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            _event, frame = _missing_call_id()
            return {"accepted": False, "ok": False, **frame}

        if call_id in calls or call_id in pending_results:
            # Already in flight or sitting unacked. The host will pick
            # the result up via SIO (either fresh emit or `resume`).
            return {"accepted": True, "call_id": call_id}

        task = _start_call(payload, call_id)

        timeout_s = max(wait_s, 0.0)
        try:
            event, frame = await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
        except TimeoutError:
            task.add_done_callback(
                lambda t, cid=call_id: asyncio.create_task(_emit_task_result(t, cid))
            )
            return {"accepted": True, "call_id": call_id}

        if event == "call:result":
            return {"accepted": False, "ok": True, **frame}
        return {"accepted": False, "ok": False, **frame}

    # Runtime internal hook used by the HTTP fast-path endpoint.
    setattr(sio, "submit_http_call", submit_http_call)

    async def on_connect(sid: str, environ: dict, auth: Any = None) -> None:
        logger.debug("sio connect %s", sid)

    async def on_disconnect(sid: str) -> None:
        # Tasks intentionally outlive the connection. Their results
        # land in `pending_results` and will be replayed on the next
        # `resume`. The host may also cancel explicitly via `cancel`.
        logger.debug("sio disconnect %s", sid)

    # ── RPC on `/rpc` ────────────────────────────────────────────

    async def on_call(sid: str, data: Any) -> None:
        payload = _u(data)
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            event, frame = _missing_call_id()
            await sio.emit(
                event,
                pack(frame),
                to=sid,
                namespace=RPC_NAMESPACE,
            )
            return

        if call_id in calls or call_id in pending_results:
            # Already running, or already completed and awaiting ack.
            # `on_resume` is the path that delivers cached results.
            return

        task = _start_call(payload, call_id)
        task.add_done_callback(
            lambda t, cid=call_id: asyncio.create_task(_emit_task_result(t, cid))
        )

    async def on_cancel(sid: str, data: Any) -> None:
        payload = _u(data)
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            return
        # Cancel is a terminal explicit signal: drop the task AND any
        # cached unacked result so the call_id is fully retired.
        pending_results.pop(call_id, None)
        call = calls.pop(call_id, None)
        if call is None:
            return
        call.cancel()
        await sio.emit(
            "call:error",
            pack(_cancelled_error(call_id)),
            to=sid,
            namespace=RPC_NAMESPACE,
        )

    async def on_resume(sid: str, data: Any) -> None:
        """Replay cached results for the call_ids the host is still
        waiting on. Called by the host right after (re)connect."""
        payload = _u(data)
        ids = payload.get("call_ids")
        if not isinstance(ids, list):
            return
        for cid in ids:
            if not isinstance(cid, str):
                continue
            cached = pending_results.get(cid)
            if cached is None:
                continue
            event, frame = cached
            await sio.emit(event, pack(frame), to=sid, namespace=RPC_NAMESPACE)

    async def on_ack(sid: str, data: Any) -> None:
        """Host confirms it has consumed the result. Free the slot."""
        payload = _u(data)
        call_id = payload.get("call_id")
        if isinstance(call_id, str):
            pending_results.pop(call_id, None)

    # ── dynamic namespace forwarding ─────────────────────────────
    #
    # `sio_open`  — worker tells us a namespace exists; we register a
    #               catch-all SIO handler that forwards every inbound
    #               event on that namespace back to the worker.
    # `sio_emit`  — worker wants to emit an event on a namespace.

    async def _on_worker_sio_frame(frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        namespace = frame.get("namespace")
        if not isinstance(namespace, str) or not namespace.startswith("/"):
            return
        if kind == "sio_emit":
            event = frame.get("event")
            if not isinstance(event, str):
                return
            await sio.emit(event, pack(frame.get("data")), namespace=namespace)
        elif kind == "sio_open":
            if namespace in opened_namespaces or namespace in {"/", RPC_NAMESPACE}:
                return
            opened_namespaces.add(namespace)
            _register_namespace(namespace)

    def _register_namespace(namespace: str) -> None:
        """Register a SIO server namespace that forwards every inbound
        event back to the worker via the pipe."""

        class _Forwarder(socketio.AsyncNamespace):
            async def trigger_event(self, event: str, *args: Any) -> Any:
                # Skip lifecycle events (connect/disconnect/connect_error)
                # — those are SIO-internal, not user-emitted data.
                if event in ("connect", "disconnect", "connect_error"):
                    return
                # args = (sid, data?)  — server namespaces pass sid first.
                data = _decode(args[1]) if len(args) >= 2 else None
                await worker.send_inbound(namespace, event, data)

        sio.register_namespace(_Forwarder(namespace))

    worker.set_sio_handler(_on_worker_sio_frame)

    # Register RPC handlers on `/rpc` non-decorator-style — `@sio.on(name)`
    # decorates by side effect and pyright can't tell that the wrapped
    # function is still usable.
    sio.on("connect", on_connect, namespace=RPC_NAMESPACE)
    sio.on("disconnect", on_disconnect, namespace=RPC_NAMESPACE)
    sio.on("call", on_call, namespace=RPC_NAMESPACE)
    sio.on("cancel", on_cancel, namespace=RPC_NAMESPACE)
    sio.on("resume", on_resume, namespace=RPC_NAMESPACE)
    sio.on("ack", on_ack, namespace=RPC_NAMESPACE)

    # Pre-register core namespaces so the host can connect to them
    # immediately — the worker subscribes lazily, but the SIO server
    # must already accept the connection.
    for core_ns in ("/trace", "/log"):
        opened_namespaces.add(core_ns)
        _register_namespace(core_ns)

    asgi_app = socketio.ASGIApp(sio, socketio_path="/socket.io")
    return sio, asgi_app
