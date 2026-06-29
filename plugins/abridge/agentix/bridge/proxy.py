"""abridge core: in-sandbox HTTP tunnel + host-side `Proxy` dispatcher.

```
agent  ──(HTTP POST /<path>)──▶  sandbox tunnel (this module, sandbox side)
                                    │ SIO event = path; payload = body
                                    ▼
                                 host  ──▶  Proxy (this module, host side)
                                               │  finds the matching @on(path)
                                               │  method on a user `client` and
                                               │  awaits it
                                               ▼
                                           user code   ──▶  upstream service
                                                              ▲
                                                              │  ClientResponse
                                    ◀───────────────────────  │  (bytes + media_type)
agent  ◀──(HTTP response)──  sandbox tunnel writes back verbatim
```

abridge's core is schema-agnostic: it accepts whitelisted HTTP POST paths,
decodes each JSON-object body, and dispatches that object to a host handler.
It is not an HTTP-transparent proxy — method, query, headers, non-object
JSON, and original JSON encoding are not carried across the tunnel. *What*
to do with a decoded request (translate, route, replay, mock) lives in
user-supplied handler classes via `@on(path)`-decorated methods. The bundled
`clients/` package ships LLM forwarders (`OpenAIClient`,
`AnthropicClient`, …), but the same machinery handles any HTTP
protocol that fits the request/response shape — e.g., MCP forwarding
through a single `@on("/mcp")` that dispatches on the JSON-RPC body's
`method` field, a webhook receiver, a custom RPC.

Two pieces in one file because they share the wire protocol:

  * **Sandbox side**: `_start_tunnel` / `_stop_tunnel` — async functions
    invoked via `sandbox.remote(...)`. They run a FastAPI app that
    listens on `127.0.0.1:<port>`, registers one POST route per declared
    path (whitelist; nothing else gets through), and SIO-emits each
    captured request on an event named after the path.
  * **Host side**: `Proxy(AsyncClientNamespace)` — the user-facing
    object. Discovers `@on(path)` methods on any number of handler
    objects (mixins, composition — anything), wires each as the SIO
    handler for that path-named event, and exposes `start(sandbox)` /
    `stop(sandbox)` / `session(sandbox)` for tunnel lifecycle.

The two halves never share a process; they only share the path strings
that the host derives from `@on(...)` and ships to the sandbox at
`start`.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import socket
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import uvicorn
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, Response

import agentix
from agentix import AsyncClientNamespace, RemoteSioError, Sandbox
from agentix.runtime.shared.codec import unpack as _msgpack_unpack

logger = logging.getLogger(__name__)

NAMESPACE = "/abridge"


# ── wire types ───────────────────────────────────────────────────────────


class AbridgeError(Exception):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class Request:
    """One inbound HTTP call captured by the sandbox tunnel.

    `path` is the URL path the agent hit (matches the `@on(...)` value
    that routed this request); `body` is the decoded JSON object. Nothing
    else is retained — the tunnel drops the HTTP method, query, headers,
    non-object JSON, and source encoding. Rollout identity (session_id,
    per-call request_id) lives on the `Client` instance, which may add
    `x-session-id` / `x-request-id` to an upstream call itself.
    """

    path: str
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClientResponse:
    """What an `@on` handler returns to the bridge.

    The buffered `body` is written to the agent's HTTP response with
    `media_type` as the Content-Type and `status_code` as the HTTP status.
    Two constructors cover the common cases:

      * `ClientResponse.json(dict)` — plain JSON response.
      * `ClientResponse.sse(bytes, ...)` — a complete, pre-rendered SSE
        blob for SDK compatibility. It is buffered, not chunk-streamed.
    """

    body: bytes
    media_type: str = "application/json"
    status_code: int = 200

    @classmethod
    def json(cls, body: dict[str, Any], *, status_code: int = 200) -> ClientResponse:
        return cls(
            body=json.dumps(body).encode(),
            media_type="application/json",
            status_code=status_code,
        )

    @classmethod
    def sse(cls, body: bytes, *, status_code: int = 200) -> ClientResponse:
        return cls(body=body, media_type="text/event-stream", status_code=status_code)


# ── handler type + Client marker + decorator ────────────────────────────


# A handler is an `async def fn(self, request: Request) -> ClientResponse`
# method on a user class, bound to its instance when `Proxy` collects it.
# `Handler` is the bound shape (no `self`).
Handler = Callable[[Request], Awaitable[ClientResponse]]


class Client(Protocol):
    """Marker protocol for a handler object passed to `Proxy(...)`.

    Two kinds qualify, and they compose: a class with at least one
    `@on(path)`-decorated method, and/or a class implementing
    `DynamicRoutes.abridge_routes()` (paths chosen at construction, e.g.
    `Forward`). The protocol requires nothing structurally — `@on` is a
    method-level attribute, not a class-level signature — so it is deliberately
    NOT `runtime_checkable`: `isinstance(x, Client)` would be true for anything
    and is meaningless. Handler presence is validated by `Proxy.__init__` at
    construction time. The name just makes `Proxy(*clients: Client)` read as
    "pass handler objects here". A client may additionally implement async
    `aclose()`; `Proxy.stop()` closes such clients once per lifecycle.
    """


@runtime_checkable
class DynamicRoutes(Protocol):
    """A client that contributes routes chosen at construction time — paths the
    class-level `@on` tag can't express, e.g. `Forward(target, paths=[...])`.

    `abridge_routes()` returns `{path: handler}`; `Proxy` merges it alongside any
    `@on` handlers under the same duplicate-path rule. This is the typed, blessed
    second registration seam (vs. `@on`): a handler returning dynamic routes has a
    checkable contract instead of an undocumented duck-typed method.
    """

    def abridge_routes(self) -> dict[str, Handler]: ...


@runtime_checkable
class _AsyncCloseable(Protocol):
    def aclose(self) -> Awaitable[None]: ...


# Preserves the decorated method's exact signature for IDE / pyright —
# `@on(path)` is otherwise a pure side-effect (it tags the function with
# an attribute), so callers and overriders see the unchanged signature.
_F = TypeVar("_F", bound=Callable[..., Awaitable[ClientResponse]])

_ON_ATTR = "_abridge_path"


def on(path: str) -> Callable[[_F], _F]:
    """Mark a method as the handler for HTTP `path`.

    The decorated method must be `async def fn(self, request: Request) ->
    ClientResponse`. abridge's `Proxy` walks the client's MRO at
    construction, collects every `@on(path)`-decorated method, and routes
    SIO events named after the path to the matching method.

    The same `path` registered twice (e.g. by two different mixins) is a
    construction-time `ValueError`. Two mixins handling unrelated paths
    compose freely — that's the whole point.

    The decorator also wraps the call with per-invocation logging:
    DEBUG on entry, INFO on success with elapsed-ms + response status.
    Errors propagate unlogged here — `Proxy._dispatch_request` owns the
    wire-level error logging so the two layers don't double up.
    """
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"@on(path) requires a path that starts with '/'; got {path!r}")

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        async def wrapper(self: Any, request: Request) -> ClientResponse:
            logger.debug("abridge → %s", path)
            t0 = time.perf_counter()
            response = await fn(self, request)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "abridge ✓ %s in %.1fms (status=%d, %s)",
                path, elapsed_ms, response.status_code, response.media_type,
            )
            return response

        setattr(wrapper, _ON_ATTR, path)
        return cast(_F, wrapper)

    return decorator


def _collect_handlers(client: Client) -> dict[str, Handler]:
    """Walk `type(client).__mro__` for `@on(...)`-decorated methods.

    Subclass overrides win (we see them first); skipped if a same-named
    attribute was already visited. Two different methods registering the
    same path raise `ValueError`.
    """
    handlers: dict[str, Handler] = {}
    seen_names: set[str] = set()
    for cls in type(client).__mro__:
        for name, attr in vars(cls).items():
            if name in seen_names:
                continue
            seen_names.add(name)
            path = getattr(attr, _ON_ATTR, None)
            if not isinstance(path, str):
                continue
            if path in handlers:
                raise ValueError(
                    f"duplicate @on handler for path {path!r} "
                    f"({type(client).__name__})"
                )
            handlers[path] = getattr(client, name)

    # Dynamic routes: a client implementing `DynamicRoutes` contributes paths
    # chosen at construction time (e.g. `Forward(target, paths=[...])`), which
    # the class-level `@on` tag can't express. They compose with `@on` handlers
    # under the same duplicate-path rule.
    if isinstance(client, DynamicRoutes):
        routes: object = client.abridge_routes()
        if not isinstance(routes, dict):
            raise TypeError(
                f"{type(client).__name__}.abridge_routes() must return a dict[str, handler]"
            )
        for path, handler in routes.items():
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError(f"abridge_routes path must start with '/'; got {path!r}")
            if path in handlers:
                raise ValueError(
                    f"duplicate handler for path {path!r} ({type(client).__name__})"
                )
            handlers[path] = handler
    return handlers


# ── handles ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TunnelHandle:
    """What `Proxy.start(sandbox)` returns: the sandbox-loopback URL the
    agent's SDK should point at, plus the bound port for debugging."""

    url: str
    port: int


@dataclass(slots=True)
class _Running:
    server: uvicorn.Server
    task: asyncio.Task


# Sandbox-process state. Keyed by url; populated in `_start_tunnel`,
# consumed in `_stop_tunnel`.
_running: dict[str, _Running] = {}


# ── sandbox-side: HTTP tunnel ────────────────────────────────────────────


class _SandboxNamespace(agentix.Namespace):
    """SIO namespace the sandbox tunnel uses to talk to the host."""

    namespace = NAMESPACE


_namespace_singleton: _SandboxNamespace | None = None


def _get_namespace() -> _SandboxNamespace:
    global _namespace_singleton
    if _namespace_singleton is None:
        _namespace_singleton = _SandboxNamespace()
        agentix.register_namespace(_namespace_singleton)
    return _namespace_singleton


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


async def _start_tunnel(
    *,
    paths: list[str],
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout: float = 600.0,
) -> TunnelHandle:
    """Sandbox entrypoint. Boots the FastAPI tunnel with one POST route
    per path in the whitelist; everything else 404s."""
    ns = _get_namespace()
    app = FastAPI()

    @app.get("/_health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    for path in paths:
        app.post(path)(_make_forwarder(
            ns=ns, path=path, request_timeout=request_timeout,
        ))

    bound_port = port or _free_port(host)
    # `ws="none"`: tunnel handles HTTP POST only. Disabling WebSocket
    # detection skips uvicorn's import of `websockets.legacy` (which is
    # deprecated noise we don't need).
    config = uvicorn.Config(app, host=host, port=bound_port, log_level="warning", ws="none")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await _wait_uvicorn_started(server)

    url = f"http://{host}:{bound_port}"
    _running[url] = _Running(server=server, task=task)
    logger.info("abridge tunnel listening on %s (paths=%s)", url, paths)
    return TunnelHandle(url=url, port=bound_port)


async def _stop_tunnel(*, handle: TunnelHandle) -> None:
    rec = _running.pop(handle.url, None)
    if rec is None:
        return
    rec.server.should_exit = True
    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(rec.task, timeout=5)


async def _wait_uvicorn_started(server: uvicorn.Server) -> None:
    for _ in range(200):
        if getattr(server, "started", False):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("uvicorn did not bind in time")


def _make_forwarder(
    *,
    ns: _SandboxNamespace,
    path: str,
    request_timeout: float,
) -> Callable[[FastAPIRequest], Awaitable[Response]]:
    """One FastAPI handler per path. Closes over `path` so the SIO event
    name (= path) is fixed per route.

    The wire payload is the decoded JSON-object body — no HTTP metadata.
    Identity (session_id / record_id) belongs to the host-side `Client`,
    which may stamp `x-session-id` / `x-request-id` on an upstream HTTP
    call. This tunnel does not preserve the original request bytes.
    """

    async def forward(request: FastAPIRequest) -> Response:
        body = await _read_json(request)
        if body is None:
            # Body was present but not a JSON object (unparseable or a non-object
            # like an array/string). Fail at the boundary with a precise error
            # instead of silently coercing to {} and confusing the upstream.
            return JSONResponse(
                {"error": {"message": "request body must be a JSON object"}},
                status_code=400,
            )

        try:
            # SIO event name IS the path; the host's `Proxy` has a
            # matching handler registered under the same name. The wire
            # payload is just the decoded object — no wrapping envelope
            # beyond request correlation and no HTTP metadata.
            # Single timeout source: thread the configured value into
            # `ns.request` itself. A redundant outer `wait_for` here was a no-op
            # above `ns.request`'s own (smaller) default, silently capping the
            # configurable `request_timeout` at that default.
            result = await ns.request(path, body, timeout=request_timeout)
        except TimeoutError:
            message = "tunnel timed out waiting for host"
            logger.warning("abridge tunnel %s: %s", path, message)
            return JSONResponse({"error": {"message": message}}, status_code=504)
        except RemoteSioError as exc:
            status = _status_from_remote_error(exc)
            logger.warning(
                "abridge tunnel %s: host raised %s: %s", path, exc.type, exc.message,
            )
            return JSONResponse({"error": {"message": exc.message}}, status_code=status)

        return _to_http_response(result)

    return forward


async def _read_json(request: FastAPIRequest) -> dict[str, Any] | None:
    """Decode the body as a JSON object. `{}` for a genuinely empty body; `None`
    when the body is present but not a JSON object (unparseable, or a valid
    non-object like an array/string). The caller turns `None` into a 400 so the
    failure surfaces at the boundary instead of as a silently-coerced `{}`."""
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _to_http_response(result: object) -> Response:
    if not isinstance(result, dict):
        raise RuntimeError(f"abridge host returned non-dict result: {result!r}")
    body_bytes = result.get("body") or b""
    if not isinstance(body_bytes, (bytes, bytearray, memoryview)):
        raise RuntimeError("abridge host returned non-bytes body")
    media_type = str(result.get("media_type") or "application/json")
    status_code = int(result.get("status_code", 200))
    return Response(content=bytes(body_bytes), media_type=media_type, status_code=status_code)


def _status_from_remote_error(exc: RemoteSioError) -> int:
    """Use the upstream HTTP status the host handler chose. `AbridgeError`
    carries `status_code` (429/400/404/...), which the host threads through the
    wire error envelope and the sandbox preserves on `RemoteSioError`. Fall back
    to 502 only when the remote error carried no status."""
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else 502


# ── host-side: Proxy ─────────────────────────────────────────────────────


class Proxy(AsyncClientNamespace):
    """Host-side abridge: SIO namespace + sandbox-tunnel lifecycle +
    `@on(path)`-based dispatch to one or more "client" objects.

    Construction discovers handlers by walking `type(client).__mro__` for
    `@on(path)`-decorated methods. Pass any number of clients (variadic):
    a single mixin-composed object, or multiple independent handler
    objects whose paths must not collide.

    ```python
    # Single object (mixin-composed handlers):
    class MyClient(SomeHandlerMixin, AnotherHandlerMixin):
        ...
    proxy = Proxy(MyClient(...))

    # Or compose at the constructor (multiple handler objects):
    proxy = Proxy(SomeClient(...), AnotherClient(...))

    async with proxy.session(sandbox) as handle:
        await sandbox.remote(agent, base_url=handle.url, ...)
    ```

    `session(sandbox)` is the recommended entry: registers the host
    namespace, starts the in-sandbox tunnel, yields a `TunnelHandle`,
    then stops the tunnel and closes clients that implement `aclose()`.
    Use `start/stop` when you need explicit lifecycle control. abridge
    core does no tracing — open
    your own `trace.span(...)` around the session if you want OTel
    grouping; the bundled clients populate it via `populate_*_span`.
    """

    def __init__(self, *clients: Client) -> None:
        super().__init__(NAMESPACE)
        if not clients:
            raise ValueError("Proxy requires at least one client with @on-decorated handlers")
        self._handle: TunnelHandle | None = None
        self._clients = clients
        self._clients_closed = False

        handlers: dict[str, Handler] = {}
        for client in clients:
            for path, method in _collect_handlers(client).items():
                if path in handlers:
                    raise ValueError(
                        f"two clients register the same @on path {path!r}"
                    )
                handlers[path] = method
        if not handlers:
            raise ValueError(
                "no @on-decorated handlers found on any client passed to Proxy(...)"
            )
        self._handlers: dict[str, Handler] = handlers
        self.paths: tuple[str, ...] = tuple(handlers)

    # ── SIO dispatch ───────────────────────────────────────────────────
    #
    # SIO event names contain `/` so the base class's `on_<event>` attribute
    # lookup won't catch them at class-definition time — we override
    # `trigger_event` to look the event up in `self._handlers` and run the
    # standard request-handler envelope dance (unwrap `request_id`/`data`,
    # call the user method, emit `<path>:result` or `<path>:error`).

    async def trigger_event(self, event: str, *args: Any) -> Any:
        if event in ("connect", "disconnect", "connect_error"):
            return await super().trigger_event(event, *args)
        method = self._handlers.get(event)
        if method is None:
            return await super().trigger_event(event, *args)

        # Decode msgpack payload exactly like the base does, then run the
        # handler detached so a slow upstream call doesn't block the SIO
        # receive loop. `unpack` (public, agentix.runtime.shared.codec)
        # accepts any buffer-protocol input — bytes, bytearray, memoryview.
        if args and isinstance(args[0], (bytes, bytearray, memoryview)):
            args = (_msgpack_unpack(args[0]),) + args[1:]
        payload = args[0] if args else None

        if not hasattr(self, "_detached_tasks"):
            self._detached_tasks = set()
        task = asyncio.create_task(self._dispatch_request(event, method, payload))
        self._detached_tasks.add(task)
        task.add_done_callback(self._detached_tasks.discard)
        return None

    async def _dispatch_request(
        self, path: str, method: Handler, payload: object
    ) -> None:
        # SIO envelope = `{request_id, data}` (per `Namespace.request`);
        # `data` IS the agent's request body — no further wrapping.
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        data = payload.get("data") if isinstance(payload, dict) else payload
        body = data if isinstance(data, dict) else {}
        request = Request(path=path, body=body)

        try:
            response = await method(request)
        except AbridgeError as exc:
            logger.warning(
                "abridge %s: handler raised AbridgeError: %s (status=%d)",
                path, exc.message, exc.status_code,
            )
            await self.emit(
                f"{path}:error",
                {
                    "request_id": request_id,
                    "error": {
                        "type": "AbridgeError",
                        "message": exc.message,
                        "status_code": exc.status_code,
                    },
                },
            )
            return
        except Exception as exc:  # noqa: BLE001 - any handler failure becomes a wire error
            msg = f"{type(exc).__name__}: {exc}"
            logger.exception("abridge %s: handler raised", path)
            await self.emit(
                f"{path}:error",
                {
                    "request_id": request_id,
                    "error": {"type": type(exc).__name__, "message": msg},
                },
            )
            return

        # response is `ClientResponse`. The reply event name follows the
        # `request_handler` convention so `Namespace.request(...)` resolves
        # the future cleanly on the sandbox side.
        await self.emit(
            f"{path}:result",
            {
                "request_id": request_id,
                "value": {
                    "body": response.body,
                    "media_type": response.media_type,
                    "status_code": response.status_code,
                },
            },
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    async def start(self, sandbox: Sandbox) -> TunnelHandle:
        """Register this `Proxy` as a host-side namespace, start the
        in-sandbox tunnel with `self.paths` as the whitelist, and return
        the `TunnelHandle` (loopback URL + port).

        Calling `start` again while this proxy is active returns the current
        handle instead of leaking another tunnel. If startup fails, clients
        with `aclose()` are still closed.

        ORDERING: open the proxy before any other `sandbox.remote()` / health
        call. The `/abridge` namespace must be registered before the runtime
        client connects, so `proxy.start` / `proxy.session` has to run first; a
        prior remote call that already connected the client makes this raise.
        """
        if self._handle is not None:
            return self._handle

        self._clients_closed = False
        try:
            try:
                sandbox.register_namespace(self)
            except RuntimeError as exc:
                # register_namespace only raises RuntimeError once the runtime
                # client has connected — i.e. a remote()/health() ran first.
                # Re-raise in this surface's vocabulary instead of the low-level
                # "before entering the async context" message.
                raise RuntimeError(
                    "open the abridge proxy (proxy.session/start) before any other "
                    "sandbox.remote()/health() call — its /abridge namespace must be "
                    "registered before the runtime client connects"
                ) from exc
            handle = await sandbox.remote(_start_tunnel, paths=list(self.paths))
        except BaseException:
            try:
                await self._close_clients()
            except Exception:  # noqa: BLE001 - preserve the startup failure
                logger.exception("abridge failed to close clients after tunnel startup failed")
            raise

        self._handle = handle
        return handle

    async def stop(self, sandbox: Sandbox) -> None:
        """Stop the active tunnel and close lifecycle-aware clients.

        Safe to call more than once. Client cleanup also runs when no tunnel
        was established, which makes an explicit `stop` safe after a failed
        or partially completed startup.
        """
        handle = self._handle
        self._handle = None
        if handle is None:
            await self._close_clients()
            return

        try:
            await sandbox.remote(_stop_tunnel, handle=handle)
        except BaseException:
            try:
                await self._close_clients()
            except Exception:  # noqa: BLE001 - preserve the tunnel stop failure
                logger.exception("abridge failed to close clients after tunnel stop failed")
            raise
        await self._close_clients()

    async def _close_clients(self) -> None:
        if self._clients_closed:
            return
        self._clients_closed = True

        errors: list[Exception] = []
        for client in self._clients:
            if not isinstance(client, _AsyncCloseable):
                continue
            try:
                await client.aclose()
            except Exception as exc:  # noqa: BLE001 - close every owned client
                errors.append(exc)
        if errors:
            raise ExceptionGroup("failed to close abridge clients", errors)

    @contextlib.asynccontextmanager
    async def session(self, sandbox: Sandbox) -> AsyncIterator[TunnelHandle]:
        """`start` + `stop`-on-exit sugar, including client cleanup.

        Open your own `trace.span(...)` around this CM if you want
        per-rollout grouping in OTel — abridge core leaves tracing entirely
        to its clients and to caller code.
        """
        handle = await self.start(sandbox)
        try:
            yield handle
        finally:
            # If the body is already raising, a teardown failure must not mask
            # it (the finally's exception would demote the real error to
            # __context__). Log-and-swallow stop() errors in that case; surface
            # them normally when the body succeeded.
            body_failed = sys.exc_info()[1] is not None
            try:
                await self.stop(sandbox)
            except Exception:
                if not body_failed:
                    raise
                logger.exception("abridge: session teardown failed after body error (suppressed)")

    # ── handy property ────────────────────────────────────────────────

    @property
    def url(self) -> str:
        """Loopback URL the in-sandbox agent's SDK should point at. abridge
        core stays shape-blind here — each bundled `clients.<name>` exposes
        an `env_for(handle)` (or per-SDK convention) for prefixing this
        URL the way the SDK expects."""
        if self._handle is None:
            raise RuntimeError("call await proxy.start(sandbox) (or use proxy.session) first")
        return self._handle.url


__all__ = [
    "AbridgeError",
    "Client",
    "ClientResponse",
    "DynamicRoutes",
    "Handler",
    "NAMESPACE",
    "Proxy",
    "Request",
    "TunnelHandle",
    "on",
]
