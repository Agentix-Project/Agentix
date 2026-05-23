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
records flow from the sandbox without setup. `/` carries RPC.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pickle
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal, ParamSpec, TypeVar, cast

import httpx
import socketio

from agentix.runtime.shared import MAX_MESSAGE_BYTES
from agentix.runtime.shared.callables import RemoteCallable, display_name_for
from agentix.runtime.shared.codec import pack, unpack
from agentix.runtime.shared.models import HealthResponse, RemoteError

logger = logging.getLogger("agentix.runtime.client")

P = ParamSpec("P")
R = TypeVar("R")


class RemoteCallError(RuntimeError):
    """Raised when a remote callable returns a non-ok RemoteResponse."""

    def __init__(self, display_name: str, error: RemoteError):
        super().__init__(f"{display_name}: {error.type}: {error.message}")
        self.display_name = display_name
        self.error = error


def _raise_remote_error(display_name: str, error: RemoteError):
    if error.cancelled:
        raise asyncio.CancelledError(error.message)
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

    def __init__(self, base_url: str, timeout: float = 300):
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
        self._http_sync_budget_ms = 1000

    def _register_core_namespaces(self) -> None:
        """Register agentix-core's built-in `/trace` and `/log` handlers."""
        from agentix.log._bridge import HostLogNamespace
        from agentix.trace._bridge import HostTraceNamespace

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
        r = await self._client.get("/health")
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

    async def _try_http_fast_path(
        self,
        *,
        sio: socketio.AsyncClient,
        payload: dict[str, Any],
    ) -> tuple[Literal["fallback", "accepted", "result", "error"], Any]:
        sid = getattr(sio, "sid", None)
        if not (isinstance(sid, str) and sid):
            return "fallback", None

        http_payload = {
            **payload,
            "prefer_sync_ms": self._http_sync_budget_ms,
        }
        r = await self._client.post(
            "/call",
            content=pack(http_payload),
            headers={"content-type": "application/msgpack"},
        )
        r.raise_for_status()
        reply = unpack(r.content) if r.content else {}
        if not isinstance(reply, dict):
            raise RuntimeError("invalid /call reply payload")

        if reply.get("accepted") is True:
            return "accepted", None
        if reply.get("ok") is True:
            return "result", _unpickle_value(reply.get("value"))
        if reply.get("ok") is False:
            err = RemoteError.model_validate(reply.get("error") or {})
            return "error", err
        raise RuntimeError("invalid /call reply fields")

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
        terminated = False
        try:
            # Fast path: try HTTP first for short-running calls. If the
            # call exceeds the sync budget, the server returns `accepted`
            # and completes via the normal SIO result channel.
            kind, value = await self._try_http_fast_path(sio=sio, payload=payload)
            if kind == "fallback":
                await sio.emit("call", pack(payload))
            elif kind == "result":
                terminated = True
                return cast(R, value)
            elif kind == "error":
                terminated = True
                _raise_remote_error(display_name, cast(RemoteError, value))
            while True:
                kind, data = await q.get()
                if kind == "result":
                    terminated = True
                    return cast(R, _unpickle_value(data.get("value")))
                if kind == "error":
                    err = RemoteError.model_validate(data["error"])
                    terminated = True
                    _raise_remote_error(display_name, err)
        finally:
            self._pending.pop(call_id, None)
            if not terminated:
                with contextlib.suppress(BaseException):
                    await sio.emit("cancel", pack({"call_id": call_id}))

    # ── Socket.IO connection management ─────────────────────────

    async def _ensure_sio(self) -> socketio.AsyncClient:
        if self._sio is not None and self._sio.connected:
            return self._sio
        async with self._sio_lock:
            if self._sio is not None and self._sio.connected:
                return self._sio
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

            sio.on("call:result", _on_call_result)
            sio.on("call:error", _on_call_error)

            async def _on_connect(*_args):
                # Fires on initial connect and on every reconnect. Tell
                # the server which call_ids we still expect results for;
                # any cached unacked results get replayed.
                pending_ids = list(self._pending.keys())
                if not pending_ids:
                    return
                with contextlib.suppress(BaseException):
                    await sio.emit("resume", pack({"call_ids": pending_ids}))

            async def _on_disconnect(*_args):
                # Tasks survive on the server side; results are buffered
                # until ack. We rely on socketio's auto-reconnect to come
                # back, then `_on_connect` will re-emit `resume`.
                logger.debug("sio disconnect; will resume after reconnect")

            sio.on("connect", _on_connect)
            sio.on("disconnect", _on_disconnect)

            namespaces = ["/"]
            for ns in self._namespaces:
                sio.register_namespace(ns)
                if ns.namespace not in namespaces:
                    namespaces.append(ns.namespace)

            await sio.connect(self._base_url, namespaces=namespaces)
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
            await sio.emit("ack", pack({"call_id": call_id}))


__all__ = ["RemoteCallError", "RuntimeClient"]
