"""Async client for the agentix runtime server.

Wraps:
  - typed remote-call dispatch: `RuntimeClient.remote(fn, *args, **kwargs)`,
    where `fn` is a stub function imported from a closure's Python package.
    Routing key is `fn.__module__`; result is decoded into `fn`'s return type.
  - built-in `/exec`, `/upload`, `/download`, plus `/closures` introspection.
  - log subscription: `RuntimeClient.logs()` is an `AsyncIterator[LogRecord]`
    fed by a Socket.IO `log` event stream.

Two transports underneath:
  - HTTP for unary RPC (`POST /_remote`) and runtime built-ins.
  - Socket.IO for server-streaming, bidirectional, and log subscription.

The Socket.IO connection is lazy and shared across all stream/bidi/log calls
on the same client. Per-`call_id` queue routing demultiplexes concurrent calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
    overload,
)

import httpx
import socketio
from pydantic import TypeAdapter

from agentix.dispatch import STREAM_ORIGINS
from agentix.runtime.models import (
    ClosureInfo,
    ExecRequest,
    ExecResponse,
    HealthResponse,
    LogRecord,
    RemoteError,
    RemoteRequest,
    RemoteResponse,
    TraceEvent,
    UploadResponse,
)

logger = logging.getLogger("agentix.client")

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class RemoteCallError(RuntimeError):
    """Raised when a remote closure impl returns a non-ok RemoteResponse,
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

    # ── runtime server endpoints ─────────────────────────────────

    async def health(self) -> HealthResponse:
        r = await self._client.get("/health")
        r.raise_for_status()
        return HealthResponse.model_validate(r.json())

    async def closures(self) -> list[ClosureInfo]:
        r = await self._client.get("/closures")
        r.raise_for_status()
        return [ClosureInfo.model_validate(x) for x in r.json()]

    # ── typed remote call ────────────────────────────────────────

    @overload
    def remote(
        self,
        fn: Callable[P, AsyncIterator[T]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AsyncIterator[T]: ...

    @overload
    def remote(
        self,
        fn: Callable[P, AsyncGenerator[T, Any]],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AsyncIterator[T]: ...

    @overload
    def remote(
        self,
        fn: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Coroutine[Any, Any, R]: ...

    def remote(self, fn, *args, **kwargs):
        """Execute `fn` in the sandbox and return its typed result.

        Polymorphic on the stub's signature:
        - Both input and output are `AsyncIterator[T]` → bidi via Socket.IO;
          returns an `AsyncIterator[T]`. Caller passes its input iterator as
          the matching positional/keyword arg.
        - Output is `AsyncIterator[T]`, no input streams → server stream via
          Socket.IO; returns an `AsyncIterator[T]`.
        - Otherwise → unary HTTP; returns a coroutine resolving to the typed
          value; use `await c.remote(fn, ...)`.
        """
        sig = inspect.signature(fn)
        output_is_stream = get_origin(sig.return_annotation) in STREAM_ORIGINS
        input_is_stream = any(
            get_origin(p.annotation) in STREAM_ORIGINS
            for p in sig.parameters.values()
            if p.annotation is not inspect.Parameter.empty
        )
        if output_is_stream and input_is_stream:
            return self._remote_bidi(fn, sig, *args, **kwargs)
        if output_is_stream:
            return self._remote_stream(fn, sig, *args, **kwargs)
        return self._remote_unary(fn, sig.return_annotation, *args, **kwargs)

    async def _remote_unary(self, fn, return_ann, *args, **kwargs):
        package = fn.__module__
        method = fn.__name__
        sig = inspect.signature(fn)
        body = RemoteRequest(
            package=package, method=method,
            args=_encode_args(sig, args), kwargs=_encode_kwargs(sig, kwargs),
        )
        r = await self._client.post("/_remote", json=body.model_dump())
        r.raise_for_status()
        resp = RemoteResponse.model_validate(r.json())
        if not resp.ok:
            assert resp.error is not None
            raise RemoteCallError(package=package, method=method, error=resp.error)
        if return_ann is inspect.Signature.empty:
            return resp.value
        return TypeAdapter(return_ann).validate_python(resp.value)

    async def _remote_stream(self, fn, sig, *args, **kwargs):
        package = fn.__module__
        method = fn.__name__
        sio = await self._ensure_sio()
        call_id = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self._pending[call_id] = q

        ret_args = get_args(sig.return_annotation)
        item_adapter = TypeAdapter(ret_args[0] if ret_args else Any)
        try:
            await sio.emit("stream", {
                "call_id": call_id,
                "package": package,
                "method": method,
                "args": _encode_args(sig, args),
                "kwargs": _encode_kwargs(sig, kwargs),
            })
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
                await sio.emit("cancel", {"call_id": call_id})

    async def _remote_bidi(self, fn, sig, *args, **kwargs):
        """Bidi over Socket.IO: client emits `bidi:start`, streams inputs as
        `bidi:in`, signals end-of-input with `bidi:end_in`. Server replies via
        `bidi:out` / `bidi:end` / `bidi:error` correlated by `call_id`.
        """
        package = fn.__module__
        method = fn.__name__

        # Identify the input-stream param.
        stream_param: str | None = None
        in_item_type: Any = Any
        for pname, param in sig.parameters.items():
            if get_origin(param.annotation) in STREAM_ORIGINS:
                stream_param = pname
                in_args = get_args(param.annotation)
                in_item_type = in_args[0] if in_args else Any
                break
        if stream_param is None:
            raise TypeError(f"{package}.{method}: signature has no AsyncIterator parameter")

        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        input_iter = bound.arguments.pop(stream_param, None)
        if input_iter is None or not hasattr(input_iter, "__aiter__"):
            raise TypeError(
                f"{package}.{method}: argument '{stream_param}' must be an "
                f"AsyncIterator (got {type(input_iter).__name__})"
            )

        non_stream_kwargs = _encode_kwargs(sig, dict(bound.arguments))
        out_args = get_args(sig.return_annotation)
        out_adapter = TypeAdapter(out_args[0] if out_args else Any)
        in_adapter = TypeAdapter(in_item_type)

        sio = await self._ensure_sio()
        call_id = uuid.uuid4().hex
        q: asyncio.Queue = asyncio.Queue()
        self._pending[call_id] = q

        await sio.emit("bidi:start", {
            "call_id": call_id, "package": package, "method": method,
            "args": [], "kwargs": non_stream_kwargs,
        })

        async def _sender() -> None:
            try:
                async for item in input_iter:
                    encoded = in_adapter.dump_python(item, mode="json")
                    await sio.emit("bidi:in", {"call_id": call_id, "item": encoded})
                await sio.emit("bidi:end_in", {"call_id": call_id})
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
                await sio.emit("cancel", {"call_id": call_id})

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
                await sio.emit("logs:subscribe", payload)
            while True:
                data = await sub_q.get()
                yield LogRecord.model_validate(data)
        finally:
            self._log_subscribers.discard(sub_q)
            if not self._log_subscribers:
                with contextlib.suppress(BaseException):
                    await sio.emit("logs:unsubscribe", {})

    async def traces(
        self,
        *,
        kind: str | None = None,
        call_id: str | None = None,
    ) -> AsyncIterator[TraceEvent]:
        """Subscribe to the runtime's trace stream.

        Yields a `TraceEvent` for every `agentix.trace.emit(...)` from any
        closure. Optional `kind` and `call_id` filters are applied
        client-side; the server broadcasts all events to subscribers.
        Iteration ends when the connection closes or the caller breaks.
        """
        sio = await self._ensure_sio()
        sub_q: asyncio.Queue = asyncio.Queue()
        self._trace_subscribers.add(sub_q)
        first_sub = len(self._trace_subscribers) == 1
        try:
            if first_sub:
                await sio.emit("traces:subscribe", {})
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
                    await sio.emit("traces:unsubscribe", {})

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

            sio.on("stream:item", _on_stream_item)
            sio.on("stream:end", _on_stream_end)
            sio.on("stream:error", _on_stream_error)
            sio.on("bidi:out", _on_bidi_out)
            sio.on("bidi:end", _on_bidi_end)
            sio.on("bidi:error", _on_bidi_error)
            sio.on("log", self._on_log)
            sio.on("trace", self._on_trace)

            await sio.connect(self._base_url)
            self._sio = sio
            return sio

    async def _dispatch_event(self, kind: str, data: dict[str, Any]) -> None:
        call_id = data.get("call_id")
        q = self._pending.get(call_id) if isinstance(call_id, str) else None
        if q is not None:
            await q.put((kind, data))

    async def _on_log(self, data: dict[str, Any]) -> None:
        for q in list(self._log_subscribers):
            q.put_nowait(data)

    async def _on_trace(self, data: dict[str, Any]) -> None:
        for q in list(self._trace_subscribers):
            q.put_nowait(data)

    # ── runtime I/O primitives (exec / upload / download) ───────

    @staticmethod
    def _exec_body(
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout: float | None,
        max_output: int | None = None,
        paths_from: list[str] | None = None,
    ) -> dict[str, Any]:
        return ExecRequest(
            command=command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            max_output=max_output,
            paths_from=paths_from,
        ).model_dump(exclude_none=True)

    async def run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        max_output: int | None = None,
        paths_from: list[str] | None = None,
    ) -> ExecResponse:
        """Buffered shell exec: run `command` and return the full captured output.

        `paths_from` prepends the `bin/` of the listed closures (by Python
        package path) to PATH for this command only. Pass `["*"]` to include
        every mounted closure.
        """
        body = self._exec_body(command, cwd, env, timeout, max_output, paths_from)
        r = await self._client.post("/exec", json=body)
        r.raise_for_status()
        return ExecResponse.model_validate(r.json())

    async def run_stream(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        paths_from: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream exec output as SSE events.

        Yields decoded event dicts like:
            {"event": "stdout", "stream": "stdout", "data": "..."}
            {"event": "exit",   "exit_code": 0}
        """
        body = self._exec_body(command, cwd, env, timeout, paths_from=paths_from)
        buf = b""
        async with self._client.stream(
            "POST", "/exec", json=body, headers={"accept": "text/event-stream"}
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                buf += chunk
                while b"\n\n" in buf:
                    event_bytes, buf = buf.split(b"\n\n", 1)
                    event = _parse_sse_event(event_bytes)
                    if event is not None:
                        yield event

    async def upload(self, local_path: str | Path, dest: str) -> UploadResponse:
        """Upload a local file to `dest` inside the sandbox."""
        p = Path(local_path)
        with open(p, "rb") as f:
            r = await self._client.post(
                "/upload",
                files={"file": (p.name, f)},
                data={"path": dest},
            )
        r.raise_for_status()
        return UploadResponse.model_validate(r.json())

    async def download(self, path: str, local_path: str | Path) -> int:
        """Stream a sandbox file down to `local_path`."""
        r = await self._client.get("/download", params={"path": path})
        r.raise_for_status()
        lp = Path(local_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_bytes(r.content)
        return len(r.content)


def _encode_args(sig: inspect.Signature, args: tuple) -> list[Any]:
    """Encode positional args via each parameter's TypeAdapter so dataclasses,
    BaseModels and other pydantic-known types serialise to JSON-compatible
    structures. Falls back to the raw value if no annotation is available.
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
            out.append(TypeAdapter(ann).dump_python(v, mode="json"))
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
            out[k] = TypeAdapter(ann).dump_python(v, mode="json")
        except Exception:
            out[k] = v
    return out


def _parse_sse_event(raw: bytes) -> dict[str, Any] | None:
    """Parse a single SSE event block into a dict. Returns None for keepalives."""
    event: str | None = None
    data_lines: list[str] = []
    for line in raw.decode(errors="replace").splitlines():
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    payload = "\n".join(data_lines)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = {"data": payload}
    if event:
        parsed.setdefault("event", event)
    return parsed
