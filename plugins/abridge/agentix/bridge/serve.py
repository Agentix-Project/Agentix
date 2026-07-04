"""Serve abridge clients over plain HTTP — direct mode, no tunnel.

The `Proxy` tunnels LLM traffic through the host because a sandbox may
have no network egress at all. When the sandbox *can* reach the model
serving network (an in-cluster vLLM, a private gateway), routing every
call through the host adds two hops and one Python process as the
fan-in for every concurrent rollout. This module serves the same
`@on(path)` handler objects as a standalone HTTP service instead:
deploy it next to the engine and point the agent's SDK straight at it —
the host stays out of the data path.

    agent (in sandbox) ──HTTP──▶ agentix-bridge-serve ──▶ engine
                                 translation + identity stamping

Rollout identity without the host: mint a fresh placeholder API key per
rollout (the key you already inject into the sandbox) and the server
maps whatever key each request carries to `session_id_for(key)` — one
server groups any number of concurrent rollouts, and the minting side
calls the same function to correlate. Agent keys are treated as
identity, never forwarded upstream; the real upstream key stays on
this server.

Trust model: the server itself is unauthenticated by default and binds
loopback unless told otherwise — expose it only to the network segment
you already trust with the engine. Sandboxes run model-generated code;
when they share a network with the server, pass `verify_key=` (CLI:
`--require-key-prefix`) so only keys your harness minted are served and
everything else gets a 401.

Anything beyond grouping (multi-backend routing, token capture) is the
full gateway's job — this is deliberately just "the tunnel without the
tunnel", and the tunnel remains the mode for sandboxes with no egress.

    OPENAI_API_KEY=EMPTY agentix-bridge-serve \
        --upstream-base-url http://vllm:8000/v1 --upstream-model qwen3-32b \
        --host 0.0.0.0 --require-key-prefix rollout-secret-
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, Response

from .proxy import AbridgeError, Client, Handler, Request, _AsyncCloseable, _collect_handlers

logger = logging.getLogger(__name__)

__all__ = ["build_app", "build_session_app", "main", "session_id_for"]

# `resolve(request)` yields the bound `@on` method for the request —
# fixed in `build_app`, per-caller (with in-flight tracking) in
# `build_session_app`.
Resolver = Callable[[FastAPIRequest], AbstractAsyncContextManager[Handler]]


def session_id_for(caller_key: str) -> str:
    """The session id the server derives from an agent's API key.

    `sha256(key)` hex truncated to 24 chars; the empty key maps to
    `"anonymous"`. Public so the side minting per-rollout keys can
    compute the same id and correlate upstream `x-session-id` values
    (the raw key is never echoed into logs or upstream headers).
    """
    if not caller_key:
        return "anonymous"
    return hashlib.sha256(caller_key.encode()).hexdigest()[:24]


def build_app(*clients: Client) -> FastAPI:
    """A FastAPI app with one POST route per `@on(path)` handler.

    Requests and responses keep the tunnel's shapes: JSON-object bodies
    in, the handler's `ClientResponse` out, in-band JSON error bodies.
    One deliberate improvement over the tunnel wire: an `AbridgeError`'s
    status code reaches the agent here, where the tunnel's SIO leg
    collapses handler errors to 502. Other exceptions are a 502. All
    requests share the clients' sessions — for per-caller sessions use
    `build_session_app`.
    """
    handlers: dict[str, Handler] = {}
    for client in clients:
        for path, method in _collect_handlers(client).items():
            if path in handlers:
                raise ValueError(f"two clients register the same @on path {path!r}")
            handlers[path] = method
    if not handlers:
        raise ValueError("no @on-decorated handlers found on any client passed to build_app(...)")

    app = FastAPI()
    app.get("/_health")(_health)
    for path, method in handlers.items():
        app.post(path)(_make_endpoint(path, _fixed_resolver(method)))
    return app


@dataclass
class _Session:
    client: Client
    handlers: dict[str, Handler]
    refs: int = 0
    evicted: bool = False


@dataclass
class _SessionTable:
    """Caller-keyed client sessions with in-flight-safe LRU eviction.

    Eviction past `max_sessions` removes the least-recent entry from
    the table immediately (the table stays bounded), but a client is
    closed only once its in-flight requests drain — a live upstream
    call is never killed by another caller's arrival.
    """

    factory: Callable[[str], Client]
    max_sessions: int
    verify_key: Callable[[str], bool] | None
    sessions: OrderedDict[str, _Session] = field(default_factory=OrderedDict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @asynccontextmanager
    async def handler_for(self, request: FastAPIRequest, path: str) -> AsyncIterator[Handler]:
        session = await self._acquire(request)
        try:
            yield session.handlers[path]
        finally:
            await self._release(session)

    async def _acquire(self, request: FastAPIRequest) -> _Session:
        caller_key = _caller_key(request)
        if self.verify_key is not None and not self.verify_key(caller_key):
            raise _UnknownKey
        session_id = session_id_for(caller_key)

        async with self.lock:
            session = self.sessions.get(session_id)
            if session is not None:
                return self._checkout(session_id, session)

        # Build outside the lock: factory work is synchronous (SSL
        # context, SDK pool setup) and must not stall other requests.
        client = await asyncio.to_thread(self.factory, session_id)
        fresh = _Session(client=client, handlers=_collect_handlers(client))

        loser: Client | None = None
        async with self.lock:
            session = self.sessions.get(session_id)
            if session is not None:
                loser = fresh.client
            else:
                self.sessions[session_id] = fresh
                session = fresh
            session = self._checkout(session_id, session)
            evicted = self._evict_over_cap()
        if loser is not None:
            await _close_client(loser)
        for old in evicted:
            await _close_client(old)
        return session

    def _checkout(self, session_id: str, session: _Session) -> _Session:
        session.refs += 1
        self.sessions.move_to_end(session_id)
        return session

    def _evict_over_cap(self) -> list[Client]:
        """Pop least-recent entries past the cap (lock held); return the
        clients already idle and safe to close now — busy ones close
        when their last in-flight request releases."""
        idle: list[Client] = []
        while len(self.sessions) > self.max_sessions:
            _, old = self.sessions.popitem(last=False)
            old.evicted = True
            if old.refs == 0:
                idle.append(old.client)
        return idle

    async def _release(self, session: _Session) -> None:
        async with self.lock:
            session.refs -= 1
            close_now = session.evicted and session.refs == 0
        if close_now:
            await _close_client(session.client)

    async def close_all(self) -> None:
        async with self.lock:
            drained = [session.client for session in self.sessions.values()]
            self.sessions.clear()
        for client in drained:
            await _close_client(client)


def build_session_app(
    client_factory: Callable[[str], Client],
    *,
    max_sessions: int = 256,
    verify_key: Callable[[str], bool] | None = None,
) -> FastAPI:
    """Like `build_app`, but with one client per caller identity.

    Each request's API key (`x-api-key`, else `Authorization: Bearer`)
    maps to `session_id_for(key)`; `client_factory(session_id)` builds
    that caller's client on first use. The session table is LRU-bounded
    at `max_sessions` with in-flight-safe eviction (an evicted client
    closes only after its live requests finish), and every remaining
    client closes on app shutdown.

    `verify_key`, when given, gates every request: it receives the raw
    caller key (`""` when the request carries none) and a falsy return
    is a 401. Without it any reachable caller is served — bind the
    server accordingly.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be >= 1")

    probe = client_factory("probe")
    paths = tuple(_collect_handlers(probe))
    if not paths:
        raise ValueError("client_factory built a client with no @on-decorated handlers")

    table = _SessionTable(factory=client_factory, max_sessions=max_sessions, verify_key=verify_key)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The probe existed only to enumerate routes at build time.
        await _close_client(probe)
        yield
        await table.close_all()

    app = FastAPI(lifespan=lifespan)
    app.get("/_health")(_health)
    for path in paths:
        app.post(path)(_make_endpoint(path, _session_resolver(table, path)))
    return app


def _session_resolver(table: _SessionTable, path: str) -> Resolver:
    def resolve(request: FastAPIRequest) -> AbstractAsyncContextManager[Handler]:
        return table.handler_for(request, path)

    return resolve


async def _health() -> dict[str, str]:
    return {"status": "ok"}


class _UnknownKey(Exception):
    """Raised when `verify_key` rejects the caller's key."""


def _fixed_resolver(method: Handler) -> Resolver:
    @asynccontextmanager
    async def resolve(_request: FastAPIRequest) -> AsyncIterator[Handler]:
        yield method

    return resolve


def _caller_key(request: FastAPIRequest) -> str:
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


async def _close_client(client: Client) -> None:
    if isinstance(client, _AsyncCloseable):
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - cleanup must not fail the live request
            logger.exception("abridge serve: failed to close client")


def _make_endpoint(path: str, resolve: Resolver):
    async def endpoint(request: FastAPIRequest) -> Response:
        body = await _read_json(request)
        try:
            async with resolve(request) as handler:
                response = await handler(Request(path=path, body=body))
        except _UnknownKey:
            return JSONResponse({"error": {"message": "unknown API key"}}, status_code=401)
        except AbridgeError as exc:
            logger.warning("abridge serve %s: %s (status=%d)", path, exc.message, exc.status_code)
            return JSONResponse({"error": {"message": exc.message}}, status_code=exc.status_code)
        except Exception as exc:  # noqa: BLE001 - any handler failure becomes a wire error
            logger.exception("abridge serve %s: handler raised", path)
            return JSONResponse({"error": {"message": f"{type(exc).__name__}: {exc}"}}, status_code=502)
        return Response(content=response.body, media_type=response.media_type, status_code=response.status_code)

    return endpoint


async def _read_json(request: FastAPIRequest) -> dict:
    try:
        parsed = await request.json()
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main(argv: list[str] | None = None) -> None:
    """`agentix-bridge-serve` — an Anthropic-speaking front for an
    OpenAI-compatible engine, one session per caller key."""
    parser = argparse.ArgumentParser(
        prog="agentix-bridge-serve",
        description=(
            "Serve the Anthropic->OpenAI translation next to an OpenAI-compatible "
            "engine (vLLM, SGLang, a gateway). Agents point ANTHROPIC_BASE_URL at "
            "this server; each distinct agent API key becomes its own session."
        ),
    )
    parser.add_argument(
        "--upstream-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible endpoint, e.g. http://vllm:8000/v1 (env: OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--upstream-api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="real key for the upstream; never the keys agents send (env: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--upstream-model",
        default=os.environ.get("UPSTREAM_MODEL"),
        help="pin every upstream call to this model id (env: UPSTREAM_MODEL)",
    )
    parser.add_argument(
        "--require-key-prefix",
        default=os.environ.get("ABRIDGE_KEY_PREFIX"),
        help=(
            "serve only requests whose API key starts with this secret prefix; "
            "mint rollout keys as <prefix><nonce>. Unset = no auth: any reachable "
            "caller is served (env: ABRIDGE_KEY_PREFIX)"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address; expose beyond loopback deliberately")
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument("--upstream-timeout", type=float, default=180.0)
    parser.add_argument(
        "--upstream-max-retries",
        type=int,
        default=0,
        help="openai SDK retries per upstream call; keep total occupancy = timeout x (1+retries) explicit",
    )
    parser.add_argument("--max-sessions", type=int, default=256)
    args = parser.parse_args(argv)
    if not args.upstream_base_url:
        parser.error("--upstream-base-url (or OPENAI_BASE_URL) is required")

    # Lazy: the translation client needs the `openai` extra.
    from .clients import AnthropicFromOpenAIClient

    def factory(session_id: str) -> AnthropicFromOpenAIClient:
        return AnthropicFromOpenAIClient(
            base_url=args.upstream_base_url,
            api_key=args.upstream_api_key,
            model=args.upstream_model,
            timeout=args.upstream_timeout,
            max_retries=args.upstream_max_retries,
            session_id=session_id,
        )

    verify_key: Callable[[str], bool] | None = None
    if args.require_key_prefix:
        prefix = str(args.require_key_prefix)

        def _has_minted_prefix(key: str) -> bool:
            return key.startswith(prefix)

        verify_key = _has_minted_prefix

    app = build_session_app(factory, max_sessions=args.max_sessions, verify_key=verify_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
