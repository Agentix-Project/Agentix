"""Async client for the agentix runtime server.

Wraps:
  - typed remote-call dispatch: `RuntimeClient.remote(fn, *args, **kwargs)`,
    where `fn` is a stub function imported from a namespace's Python package.
    Routing key is `fn.__module__`; result is decoded into `fn`'s return type.
    Shell exec / file I/O live in the `bash` and `files` primitive namespaces —
    call them via `c.remote(bash.Bash.run, ...)` and `c.remote(files.Files.upload, ...)`.
  - `/namespaces` introspection and `/health`.
  - log subscription: `RuntimeClient.logs()` is an `AsyncIterator[LogRecord]`
    fed by a Socket.IO `log` event stream; same for `RuntimeClient.traces()`.

Two transports underneath:
  - HTTP for unary RPC (`POST /_remote`).
  - Socket.IO for server-streaming, bidirectional, and log/trace subscription.

The Socket.IO connection is lazy and shared across all stream/bidi/log calls
on the same client. Per-`call_id` queue routing demultiplexes concurrent calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine, Iterator, Mapping
from contextlib import contextmanager
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    get_args,
    overload,
)

import httpx
import socketio
from pydantic import TypeAdapter

from agentix.dispatch import detect_shape
from agentix.rpc import Bidi, Channel, Stream, Unary, is_channel_annotation
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
    TRACE,
    TRACES_SUBSCRIBE,
    TRACES_UNSUBSCRIBE,
)
from agentix.runtime.models import (
    HealthResponse,
    LogRecord,
    NamespaceInfo,
    RemoteError,
    RemoteResponse,
    TraceEvent,
)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

_CLIENT_CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentix_client_call_id",
    default=None,
)


# ── Per-call hot-path caches ─────────────────────────────────────────
#
# `c.remote(fn, ...)` runs on every dispatch. Two repeated costs would
# otherwise dominate a tight RL loop:
#   * `inspect.signature(fn, eval_str=True)` — PEP 563 annotation eval
#     is ~100–500 µs per call.
#   * `TypeAdapter(annotation)` — pydantic schema compile is ~50–200 µs
#     per parameter.
# Both are deterministic in `fn` / `annotation` identity, so a process-
# wide cache is correct.

@functools.cache
def _signature_of(fn: Callable) -> inspect.Signature:
    return inspect.signature(fn, eval_str=True)


_ADAPTER_CACHE: dict[int, TypeAdapter] = {}


def _adapter_for(ann: Any) -> TypeAdapter:
    """Return a cached `TypeAdapter` for the annotation. Keyed by
    `id(ann)`: identity-stable for annotations stored on a function's
    `__annotations__`, which live as long as the function does."""
    key = id(ann)
    a = _ADAPTER_CACHE.get(key)
    if a is None:
        a = _ADAPTER_CACHE[key] = TypeAdapter(ann)
    return a


class RemoteCallError(RuntimeError):
    """Raised when a remote namespace impl returns a non-ok RemoteResponse,
    or when a stream/bidi call surfaces an `error` event from the wire."""

    def __init__(self, package: str, method: str, error: RemoteError):
        super().__init__(f"{package}.{method}: {error.type}: {error.message}")
        self.package = package
        self.method = method
        self.error = error


class RuntimeClient:
    """Async client for the agentix runtime server."""

    def __init__(self, base_url: str, timeout: float = 300):
        self._base_url = base_url
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        # Socket.IO bookkeeping — created lazily on first stream/bidi/log call.
        self._sio: socketio.AsyncClient | None = None
        self._sio_lock = asyncio.Lock()
        # call_id -> event queue. Stream and bidi share the same machinery.
        self._pending: dict[str, asyncio.Queue] = {}
        # log + trace subscribers — each subscriber has its own queue.
        self._log_subscribers: set[asyncio.Queue] = set()
        self._trace_subscribers: set[asyncio.Queue] = set()

    # ── lifecycle ────────────────────────────────────────────────

    async def close(self):
        if self._sio is not None and self._sio.connected:
            with contextlib.suppress(BaseException):
                await self._sio.disconnect()
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    @contextmanager
    def call_context(self, *, call_id: str | None = None) -> Iterator[None]:
        """Temporarily attach call metadata to typed remote calls.

        `call_id` is forwarded into `RemoteRequest.call_id`, so trace events
        emitted by namespace code inherit it. The value is stored in a
        contextvar, making it safe for concurrent asyncio tasks.
        """
        token = _CLIENT_CALL_ID.set(call_id)
        try:
            yield
        finally:
            _CLIENT_CALL_ID.reset(token)

    # ── runtime server endpoints ─────────────────────────────────

    async def health(self) -> HealthResponse:
        r = await self._client.get("/health")
        r.raise_for_status()
        return HealthResponse.model_validate(r.json())

    async def namespaces(self) -> list[NamespaceInfo]:
        r = await self._client.get("/namespaces")
        r.raise_for_status()
        return [NamespaceInfo.model_validate(x) for x in r.json()]

    # ── typed remote call ────────────────────────────────────────

    @overload
    def remote(
        self,
        fn: Callable[P, AsyncGenerator[T, Any]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Stream[T] | Bidi[Any, T]: ...

    @overload
    def remote(
        self,
        fn: Callable[P, Coroutine[Any, Any, R]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Unary[R]: ...

    def remote(self, fn, *args, **kwargs):
        """Execute `fn` in the sandbox and return its typed result.

        Returns a tagged variant whose Python protocol matches the call
        shape — `await` a `Unary`, `async for` over a `Stream` or `Bidi`.
        Generic helpers can `match` on the variant for exhaustive
        dispatch.

          * `async def f(...) -> T`                          → `Unary[T]`
          * `async def f(...) -> AsyncIterator[T]: yield ...` → `Stream[T]`
          * `async def f(..., inbox: Channel[I]) -> AsyncIterator[T]: yield ...`
            → `Bidi[I, T]`; caller pushes inputs via `inbox.send(...)`
        """
        sig = _signature_of(fn)
        shape = detect_shape(fn, sig)
        if shape == "unary":
            return Unary(self._remote_unary(fn, sig, args, kwargs))
        if shape == "stream":
            return Stream(self._remote_stream(fn, sig, args, kwargs))
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        chan_name, inbox, item_type = self._extract_channel(fn, sig, bound.arguments)
        return Bidi(inbox, self._remote_bidi(fn, sig, chan_name, item_type, inbox, bound.arguments))

    @staticmethod
    def _extract_channel(
        fn, sig: inspect.Signature, arguments: Mapping[str, Any],
    ) -> tuple[str, Channel, Any]:
        """Locate the `Channel[T]` parameter, pull the user-passed Channel
        instance from bound arguments, and recover `T` for item validation.

        Returns `(param_name, channel, item_type)`. Raises if the bidi
        method has no Channel-typed param, or the user didn't pass a
        Channel instance for it."""
        for pname, param in sig.parameters.items():
            if not is_channel_annotation(param.annotation):
                continue
            ch = arguments.get(pname)
            if not isinstance(ch, Channel):
                raise TypeError(
                    f"{fn.__module__}.{fn.__name__}: bidi parameter "
                    f"'{pname}' must be an agentix.Channel instance "
                    f"(got {type(ch).__name__})"
                )
            type_args = get_args(param.annotation)
            return pname, ch, (type_args[0] if type_args else Any)
        raise TypeError(
            f"{fn.__module__}.{fn.__name__}: bidi method has no "
            f"Channel[T] parameter"
        )

    async def _remote_unary(self, fn, sig, args, kwargs):
        package = fn.__module__
        method = fn.__name__
        payload = {
            "package": package, "method": method,
            "args": _encode_args(sig, args),
            "kwargs": _encode_kwargs(sig, kwargs),
        }
        call_id = _CLIENT_CALL_ID.get()
        if call_id is not None:
            payload["call_id"] = call_id
        body = pack(payload)
        r = await self._client.post(
            "/_remote", content=body,
            headers={"Content-Type": "application/msgpack"},
        )
        r.raise_for_status()
        resp = RemoteResponse.model_validate(unpack(r.content))
        if not resp.ok:
            assert resp.error is not None
            raise RemoteCallError(package=package, method=method, error=resp.error)
        return_ann = sig.return_annotation
        if return_ann is inspect.Signature.empty:
            return resp.value
        return _adapter_for(return_ann).validate_python(resp.value)

    async def _remote_stream(self, fn, sig, args, kwargs):
        package = fn.__module__
        method = fn.__name__
        sio = await self._ensure_sio()
        call_id = uuid.uuid4().hex
        trace_call_id = _CLIENT_CALL_ID.get()
        q: asyncio.Queue = asyncio.Queue()
        self._pending[call_id] = q

        ret_args = get_args(sig.return_annotation)
        item_adapter = _adapter_for(ret_args[0] if ret_args else Any)
        try:
            payload = {
                "call_id": call_id,
                "package": package,
                "method": method,
                "args": _encode_args(sig, args),
                "kwargs": _encode_kwargs(sig, kwargs),
            }
            if trace_call_id is not None:
                payload["trace_call_id"] = trace_call_id
            await sio.emit(STREAM, pack(payload))
            while True:
                kind, data = await q.get()
                if kind == "end":
                    return
                if kind == "error":
                    err = RemoteError.model_validate(data["error"])
                    raise RemoteCallError(package=package, method=method, error=err)
                if kind == "item":
                    yield item_adapter.validate_python(data["value"])
        finally:
            self._pending.pop(call_id, None)
            with contextlib.suppress(BaseException):
                await sio.emit(CANCEL, pack({"call_id": call_id}))

    async def _remote_bidi(
        self, fn, sig, chan_name: str, in_item_type: Any,
        inbox: Channel, bound_arguments: Mapping[str, Any],
    ):
        """Bidi over Socket.IO: client emits `bidi:start`, streams inputs as
        `bidi:in`, signals end-of-input with `bidi:end_in`. Server replies via
        `bidi:out` / `bidi:end` / `bidi:error` correlated by `call_id`.

        `chan_name`, `inbox`, and `in_item_type` are already discovered
        by `_extract_channel` in `remote()`; this method does not
        re-scan the signature.
        """
        package = fn.__module__
        method = fn.__name__

        non_channel_kwargs = dict(bound_arguments)
        non_channel_kwargs.pop(chan_name, None)
        non_channel_kwargs = _encode_kwargs(sig, non_channel_kwargs)
        out_args = get_args(sig.return_annotation)
        out_adapter = _adapter_for(out_args[0] if out_args else Any)
        in_adapter = _adapter_for(in_item_type)

        sio = await self._ensure_sio()
        call_id = uuid.uuid4().hex
        trace_call_id = _CLIENT_CALL_ID.get()
        q: asyncio.Queue = asyncio.Queue()
        self._pending[call_id] = q

        payload = {
            "call_id": call_id, "package": package, "method": method,
            "args": [], "kwargs": non_channel_kwargs,
        }
        if trace_call_id is not None:
            payload["trace_call_id"] = trace_call_id
        await sio.emit(BIDI_START, pack(payload))

        async def _sender() -> None:
            try:
                async for item in inbox:
                    encoded = in_adapter.dump_python(item, mode="python")
                    await sio.emit(BIDI_IN, pack({"call_id": call_id, "item": encoded}))
                await sio.emit(BIDI_END_IN, pack({"call_id": call_id}))
            except Exception:
                # outer loop will see an error event or close
                pass

        sender = asyncio.create_task(_sender())
        try:
            while True:
                kind, data = await q.get()
                if kind == "end":
                    return
                if kind == "error":
                    err = RemoteError.model_validate(data["error"])
                    raise RemoteCallError(package=package, method=method, error=err)
                if kind == "item":
                    yield out_adapter.validate_python(data["value"])
        finally:
            sender.cancel()
            with contextlib.suppress(BaseException):
                await sender
            self._pending.pop(call_id, None)
            with contextlib.suppress(BaseException):
                await sio.emit(CANCEL, pack({"call_id": call_id}))

    # ── log subscription ────────────────────────────────────────

    async def logs(self, *, filter: str | None = None) -> AsyncIterator[LogRecord]:
        """Subscribe to the runtime's log stream.

        Yields a `LogRecord` for every `logging` record emitted under
        the `agentix.*` logger tree (or the `filter` prefix if given).
        Iteration ends when the connection closes or the caller breaks.
        """
        sio = await self._ensure_sio()
        sub_q: asyncio.Queue = asyncio.Queue()
        self._log_subscribers.add(sub_q)
        first_sub = len(self._log_subscribers) == 1
        try:
            if first_sub:
                payload = {"filter": filter} if filter else {}
                await sio.emit(LOGS_SUBSCRIBE, pack(payload))
            while True:
                data = await sub_q.get()
                yield LogRecord.model_validate(data)
        finally:
            self._log_subscribers.discard(sub_q)
            if not self._log_subscribers:
                with contextlib.suppress(BaseException):
                    await sio.emit(LOGS_UNSUBSCRIBE, pack({}))

    async def traces(
        self,
        *,
        kind: str | None = None,
        call_id: str | None = None,
    ) -> AsyncIterator[TraceEvent]:
        """Subscribe to the runtime's trace stream.

        Yields a `TraceEvent` for every `agentix.trace.emit(...)` from any
        namespace. Optional `kind` and `call_id` filters are applied
        client-side; the server broadcasts all events to subscribers.
        Iteration ends when the connection closes or the caller breaks.
        """
        sio = await self._ensure_sio()
        sub_q: asyncio.Queue = asyncio.Queue()
        self._trace_subscribers.add(sub_q)
        first_sub = len(self._trace_subscribers) == 1
        try:
            if first_sub:
                await sio.emit(TRACES_SUBSCRIBE, pack({}))
            while True:
                data = await sub_q.get()
                if kind is not None and data.get("kind") != kind:
                    continue
                if call_id is not None and data.get("call_id") != call_id:
                    continue
                yield TraceEvent.model_validate(data)
        finally:
            self._trace_subscribers.discard(sub_q)
            if not self._trace_subscribers:
                with contextlib.suppress(BaseException):
                    await sio.emit(TRACES_UNSUBSCRIBE, pack({}))

    # ── Socket.IO connection management ─────────────────────────

    async def _ensure_sio(self) -> socketio.AsyncClient:
        if self._sio is not None and self._sio.connected:
            return self._sio
        async with self._sio_lock:
            if self._sio is not None and self._sio.connected:
                return self._sio
            sio = socketio.AsyncClient()

            async def _on_stream_item(data): await self._dispatch_event("item", data)
            async def _on_stream_end(data):  await self._dispatch_event("end", data)
            async def _on_stream_error(data): await self._dispatch_event("error", data)
            async def _on_bidi_out(data):   await self._dispatch_event("item", data)
            async def _on_bidi_end(data):   await self._dispatch_event("end", data)
            async def _on_bidi_error(data): await self._dispatch_event("error", data)

            sio.on(STREAM_ITEM, _on_stream_item)
            sio.on(STREAM_END, _on_stream_end)
            sio.on(STREAM_ERROR, _on_stream_error)
            sio.on(BIDI_OUT, _on_bidi_out)
            sio.on(BIDI_END, _on_bidi_end)
            sio.on(BIDI_ERROR, _on_bidi_error)
            sio.on(LOG, self._on_log)
            sio.on(TRACE, self._on_trace)

            await sio.connect(self._base_url)
            self._sio = sio
            return sio

    async def _dispatch_event(self, kind: str, raw: Any) -> None:
        data = _decode_payload(raw)
        call_id = data.get("call_id")
        q = self._pending.get(call_id) if isinstance(call_id, str) else None
        if q is not None:
            await q.put((kind, data))

    async def _on_log(self, raw: Any) -> None:
        data = _decode_payload(raw)
        for q in list(self._log_subscribers):
            q.put_nowait(data)

    async def _on_trace(self, raw: Any) -> None:
        data = _decode_payload(raw)
        for q in list(self._trace_subscribers):
            q.put_nowait(data)

    # Shell exec / file I/O are not in the runtime core. Mount the `bash`
    # and `files` primitive namespaces and dispatch through `c.remote(...)`.


def _encode_args(sig: inspect.Signature, args: tuple) -> list[Any]:
    """Encode positional args via each parameter's TypeAdapter so dataclasses,
    BaseModels, ndarrays, and other native Python types pass through msgpack
    cleanly (mode="python" — msgpack ext types handle ndarray + BaseModel
    natively; no JSON intermediate).
    """
    out: list[Any] = []
    params = list(sig.parameters.values())
    for i, v in enumerate(args):
        ann = (
            params[i].annotation
            if i < len(params) and params[i].annotation is not inspect.Parameter.empty
            else Any
        )
        try:
            out.append(_adapter_for(ann).dump_python(v, mode="python"))
        except Exception:
            out.append(v)
    return out


def _encode_kwargs(sig: inspect.Signature, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Like `_encode_args` but for keyword args, looked up by parameter name."""
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        param = sig.parameters.get(k)
        ann = (
            param.annotation
            if param is not None and param.annotation is not inspect.Parameter.empty
            else Any
        )
        try:
            out[k] = _adapter_for(ann).dump_python(v, mode="python")
        except Exception:
            out[k] = v
    return out


def _decode_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes):
        return unpack(raw)
    return raw

