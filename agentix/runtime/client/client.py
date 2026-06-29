"""Async client for the agentix runtime server.

User surface:

    async with RuntimeClient(url) as c:
        result = await c.remote(fn, *args, **kwargs)

For plugin integration, register a `socketio.AsyncClientNamespace`
subclass (typically `agentix.AsyncClientNamespace`) BEFORE entering
the async context:

    client = RuntimeClient(url)
    client.register_namespace(AbridgeHost(openai_client=...))
    async with client as c:
        await c.remote(abridge.start_service, ...)

Core auto-registers `/trace` and `/log` namespaces so trace + log
records flow from the sandbox without setup. `/rpc` carries RPC.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pickle
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast

import httpx
import socketio
from socketio.exceptions import ConnectionError as SioConnectionError

from agentix.runtime.client.result import Failed, Ok, Result
from agentix.runtime.shared import MAX_MESSAGE_BYTES
from agentix.runtime.shared.callables import RemoteCallable, display_name_for
from agentix.runtime.shared.codec import pack, unpack
from agentix.runtime.shared.models import HealthResponse, RemoteError
from agentix.utils import context

logger = logging.getLogger("agentix.runtime.client")

P = ParamSpec("P")
R = TypeVar("R")
RPC_NAMESPACE = "/rpc"


class RemoteCallError(RuntimeError):
    """Raised when a remote callable returns a non-ok RemoteResponse."""

    def __init__(self, display_name: str, error: RemoteError):
        message = f"{display_name}: {error.type}: {error.message}"
        if error.traceback:
            message += f"\n--- sandbox traceback ---\n{error.traceback}"
        super().__init__(message)
        self.display_name = display_name
        self.error = error


class RuntimeUnreachable(RuntimeError):
    """Raised when the runtime server cannot be reached (connect failed)."""


class CallTimeout(RuntimeError):
    """Raised when a remote call exceeds the client's `call_deadline`.

    The call is cancelled on the server before this is raised, so the
    runtime does not keep executing an abandoned call.
    """


class WorkerExited(RemoteCallError):
    """The sandbox worker subprocess died mid-call (crash / OOM-kill / exit).

    A subclass of `RemoteCallError`, so existing `except RemoteCallError`
    handlers still catch it; catch `WorkerExited` specifically to branch on
    the worker's process exit status. `returncode` is negative when the
    worker was killed by a signal — `-9` (SIGKILL) is the OOM-killer's
    signature, the most common RL/eval failure:

        try:
            await sandbox.remote(agent.run, task=task)
        except WorkerExited as exc:
            if exc.returncode == -9:
                retry_with_more_memory()
    """

    @property
    def returncode(self) -> int | None:
        return self.error.returncode


class CallCancelled(RemoteCallError):
    """The runtime reported the call as cancelled — a terminal *server-side*
    state, distinct from local `asyncio` task cancellation. Subclasses
    `RemoteCallError` so it rides the normal terminal-state path: `remote()`
    raises it and `try_remote()` surfaces it as `Failed`. (A bare
    `asyncio.CancelledError` here would escape the `Ok | Failed` sum type and
    read as local cancellation.)"""


def _raise_remote_error(display_name: str, error: RemoteError):
    if error.cancelled:
        raise CallCancelled(display_name=display_name, error=error)
    if error.type == "WorkerDied":
        raise WorkerExited(display_name=display_name, error=error)
    raise RemoteCallError(display_name=display_name, error=error)


def _decode_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes):
        return unpack(raw)
    return raw


def _unpickle_value(raw: Any) -> Any:
    return pickle.loads(raw) if raw is not None else None


class RuntimeClient:
    """Async client for the agentix runtime server."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 300,
        *,
        call_deadline: float | None = None,
    ):
        """Connect to a runtime server at `base_url`.

        `timeout` is the per-request HTTP/WebSocket timeout in seconds; raise
        it for long agent calls (e.g. `RuntimeClient(url, timeout=1800)`).

        Every call rides one transport: Socket.IO on `/rpc`. HTTP is used
        only for the `/health` probe. Reconnection uses socketio's defaults
        (on, infinite attempts, 1–5s backoff).

        `call_deadline` (seconds, None = unbounded) is the cheap catch-all
        upper bound for any single `remote(...)`: whatever the cause — worker
        hang, silent sandbox loss, network black hole — the caller gets a
        `CallTimeout` (and the call is cancelled server-side) instead of
        hanging.
        """
        self._base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        # Socket.IO bookkeeping — created lazily on first remote call.
        self._sio: socketio.AsyncClient | None = None
        self._sio_lock = asyncio.Lock()
        # call_id → queue of (kind, data) for in-flight calls.
        self._pending: dict[str, asyncio.Queue] = {}
        # Namespaces queued for registration on connect.
        self._namespaces: list[socketio.AsyncClientNamespace] = []
        self._register_core_namespaces()
        # Upper bound for any single `remote(...)` (seconds). None = no
        # deadline. The cheap catch-all so the caller never hangs.
        self._call_deadline = call_deadline

    def _register_core_namespaces(self) -> None:
        """Register agentix-core's built-in `/trace` and `/log` handlers."""
        from agentix.utils.log._bridge import HostLogNamespace
        from agentix.utils.trace._bridge import HostTraceNamespace

        self._namespaces.append(HostTraceNamespace())
        self._namespaces.append(HostLogNamespace())

    # ── lifecycle ────────────────────────────────────────────────

    async def close(self):
        if self._sio is not None and self._sio.connected:
            with contextlib.suppress(BaseException):
                await self._sio.disconnect()
        await self._client.aclose()

    async def __aenter__(self):
        await self._ensure_sio()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ── public API ───────────────────────────────────────────────

    async def health(self) -> HealthResponse:
        try:
            r = await self._client.get("/health")
        except httpx.RequestError as exc:
            # Same wrapped, URL-bearing error `remote()` raises, so the
            # designated "is it up?" probe never leaks a bare httpx error.
            raise RuntimeUnreachable(
                f"runtime server unreachable at {self._base_url}: {exc}"
            ) from exc
        r.raise_for_status()
        return HealthResponse.model_validate(r.json())

    def register_namespace(self, ns: socketio.AsyncClientNamespace) -> None:
        """Register a namespace handler. MUST be called before entering
        the async context (the connection plan is fixed at connect time).

        Pass an `agentix.AsyncClientNamespace` subclass (or stdlib
        `socketio.AsyncClientNamespace` if you handle msgpack yourself).
        """
        if self._sio is not None:
            raise RuntimeError(
                "register_namespace must be called before entering the async context",
            )
        path = getattr(ns, "namespace", None)
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError(
                f"namespace handler must declare a namespace path (got {path!r})",
            )
        for existing in self._namespaces:
            if existing.namespace == path:
                raise ValueError(f"namespace {path!r} already registered")
        self._namespaces.append(ns)

    async def remote(
        self,
        fn: Callable[P, R] | Callable[P, Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Execute `fn(*args, **kwargs)` in the sandbox and return its result."""
        display_name = display_name_for(fn)
        callable_ref = RemoteCallable._resolve(fn)
        arguments = pickle.dumps((args, kwargs))
        sio = await self._ensure_sio()
        call_id = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self._pending[call_id] = q

        payload = {
            "call_id": call_id,
            "callable": str(callable_ref),
            "arguments": arguments,
        }
        # Auto-capture the host's ambient context (baggage + propagator
        # slices, e.g. the active trace scope) and ship it alongside the
        # call. Empty context encodes to None and adds nothing.
        carrier = context.encode()
        if carrier is not None:
            payload["context"] = carrier
        terminated = False
        try:
            # `call_deadline` is the cheap catch-all: regardless of cause
            # (worker hang, silent sandbox loss, network black hole), the
            # await below cannot block past the deadline. None = no bound.
            async with asyncio.timeout(self._call_deadline):
                await sio.emit("call", pack(payload), namespace=RPC_NAMESPACE)
                while True:
                    kind, data = await q.get()
                    if kind == "result":
                        terminated = True
                        return cast(R, _unpickle_value(data.get("value")))
                    if kind == "error":
                        # Defensive: a malformed frame with no `error` payload
                        # must still resolve to a typed terminal error, not a
                        # bare KeyError escaping `remote()` / `try_remote()`.
                        raw_err = data.get("error") or {
                            "type": "MalformedError",
                            "message": "runtime sent a call:error with no error payload",
                        }
                        err = RemoteError.model_validate(raw_err)
                        terminated = True
                        _raise_remote_error(display_name, err)
        except TimeoutError:
            raise CallTimeout(
                f"remote call '{display_name}' exceeded deadline of {self._call_deadline}s"
            ) from None
        finally:
            self._pending.pop(call_id, None)
            if not terminated:
                with contextlib.suppress(BaseException):
                    await sio.emit(
                        "cancel",
                        pack({"call_id": call_id}),
                        namespace=RPC_NAMESPACE,
                    )

    async def try_remote(
        self,
        fn: Callable[P, R] | Callable[P, Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[R]:
        """Like `remote()`, but returns a `Result[R]` (`Ok | Failed`) instead
        of raising on a terminal error — for callers that branch on the
        outcome with `match`. Misuse (a non-importable callable) and
        cancellation still raise."""
        try:
            return Ok(await self.remote(fn, *args, **kwargs))
        except (RemoteCallError, CallTimeout, RuntimeUnreachable) as exc:
            return Failed(exc)

    # ── Socket.IO connection management ─────────────────────────

    async def _ensure_sio(self) -> socketio.AsyncClient:
        if self._sio is not None and self._sio.connected:
            return self._sio
        async with self._sio_lock:
            if self._sio is not None and self._sio.connected:
                return self._sio
            if self._sio is not None:
                # A handle exists but is disconnected (a transport drop mid
                # reconnect). Tear it down before building a fresh client —
                # overwriting it without disconnecting leaks its aiohttp session
                # and background reconnect task, and would leave a second live
                # `/rpc` socket once the abandoned client reconnects on its own.
                # In-flight calls are not lost: their `self._pending` entries
                # persist and the fresh client's `connect` handler re-emits
                # `resume` to recover their results.
                with contextlib.suppress(BaseException):
                    await self._sio.disconnect()
                self._sio = None
            # `max_msg_size` lifts the websocket's receive cap (engineio's
            # client rides aiohttp, default 4 MB) — large `c.remote`
            # payloads / plugin events otherwise kill the connection.
            # Matches the server's `max_http_buffer_size`.
            sio = socketio.AsyncClient(
                websocket_extra_options={"max_msg_size": MAX_MESSAGE_BYTES},
            )

            async def _on_call_result(data):
                await self._route_event("result", data)

            async def _on_call_error(data):
                await self._route_event("error", data)

            sio.on("call:result", _on_call_result, namespace=RPC_NAMESPACE)
            sio.on("call:error", _on_call_error, namespace=RPC_NAMESPACE)

            async def _on_connect(*_args):
                # Fires on initial connect and on every reconnect. Tell
                # the server which call_ids we still expect results for;
                # any cached unacked results get replayed.
                pending_ids = list(self._pending.keys())
                if not pending_ids:
                    return
                with contextlib.suppress(BaseException):
                    await sio.emit(
                        "resume",
                        pack({"call_ids": pending_ids}),
                        namespace=RPC_NAMESPACE,
                    )

            async def _on_disconnect(*_args):
                # Reconnection is on, so server-side tasks survive and
                # `_on_connect` re-emits `resume` to recover results — just
                # wait. `call_deadline` bounds the wait if it never reconnects.
                logger.debug("sio disconnect; will resume after reconnect")

            async def _on_connect_error(*args):
                # Previously unobserved: surface (re)connection failures so
                # they aren't silently dropped.
                logger.debug("sio connect_error for %s: %s", self._base_url, args[0] if args else "")

            sio.on("connect", _on_connect, namespace=RPC_NAMESPACE)
            sio.on("disconnect", _on_disconnect, namespace=RPC_NAMESPACE)
            sio.on("connect_error", _on_connect_error, namespace=RPC_NAMESPACE)

            namespaces = [RPC_NAMESPACE]
            for ns in self._namespaces:
                sio.register_namespace(ns)
                if ns.namespace not in namespaces:
                    namespaces.append(ns.namespace)

            try:
                await sio.connect(self._base_url, namespaces=namespaces)
            except (SioConnectionError, OSError) as exc:
                raise RuntimeUnreachable(
                    f"runtime server unreachable at {self._base_url}: {exc}"
                ) from exc
            self._sio = sio
            return sio

    async def _route_event(self, kind: str, raw: Any) -> None:
        data = _decode_payload(raw)
        call_id = data.get("call_id")
        if not isinstance(call_id, str):
            return
        q = self._pending.get(call_id)
        if q is None:
            # Either a duplicate replay or an event for a call_id we
            # already retired. Ack defensively so the server can free
            # the slot.
            await self._ack(call_id)
            return
        await q.put((kind, data))
        # The event has landed in this process; the server can drop it
        # from its replay buffer regardless of whether `remote()` ends
        # up returning it to user code.
        await self._ack(call_id)

    async def _ack(self, call_id: str) -> None:
        sio = self._sio
        if sio is None or not sio.connected:
            return
        with contextlib.suppress(BaseException):
            await sio.emit("ack", pack({"call_id": call_id}), namespace=RPC_NAMESPACE)


__all__ = [
    "CallTimeout",
    "RemoteCallError",
    "RuntimeClient",
    "RuntimeUnreachable",
    "WorkerExited",
]
