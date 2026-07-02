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
derives `x-session-id` from a hash of whatever key each request
carries — one server groups any number of concurrent rollouts, and the
minting side can compute the same hash to correlate. Agent keys are
treated as identity, never forwarded upstream; the real upstream key
stays on this server. Anything beyond grouping (multi-backend routing,
token capture) is the full gateway's job — this is deliberately just
"the tunnel without the tunnel", and the tunnel remains the mode for
sandboxes with no egress.

    OPENAI_API_KEY=EMPTY agentix-bridge-serve \
        --upstream-base-url http://vllm:8000/v1 --upstream-model qwen3-32b
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable

import uvicorn
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, Response

from .proxy import AbridgeError, Client, Handler, Request, _AsyncCloseable, _collect_handlers

logger = logging.getLogger(__name__)

__all__ = ["build_app", "build_session_app", "main"]

# `resolve(request) -> Handler`: fixed in `build_app`, per-caller in
# `build_session_app`.
Resolver = Callable[[FastAPIRequest], Awaitable[Handler]]


def build_app(*clients: Client) -> FastAPI:
    """A FastAPI app with one POST route per `@on(path)` handler.

    Same request/response contract as the sandbox tunnel: JSON-object
    bodies in, the handler's `ClientResponse` out, handler errors as
    JSON error bodies (`AbridgeError` keeps its status; anything else
    is a 502). All requests share the clients' sessions — for
    per-caller sessions use `build_session_app`.
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


def build_session_app(
    client_factory: Callable[[str], Client],
    *,
    max_sessions: int = 256,
) -> FastAPI:
    """Like `build_app`, but with one client per caller identity.

    Each request's API key (`x-api-key`, else `Authorization: Bearer`)
    hashes to a session id; `client_factory(session_id)` builds that
    caller's client on first use. A capped LRU keeps live clients and
    closes (via `aclose()`, when implemented) the least recent one past
    `max_sessions` — size it above your rollout concurrency so an
    in-flight caller is never evicted. Keyless requests share the
    `"anonymous"` session.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be >= 1")

    anonymous = client_factory("anonymous")
    paths = tuple(_collect_handlers(anonymous))
    if not paths:
        raise ValueError("client_factory built a client with no @on-decorated handlers")

    # session id → that caller's handler table (client kept alive by its
    # bound methods; kept alongside for aclose on eviction).
    sessions: OrderedDict[str, tuple[Client, dict[str, Handler]]] = OrderedDict()
    sessions["anonymous"] = (anonymous, _collect_handlers(anonymous))
    lock = asyncio.Lock()

    def _session_resolver(path: str) -> Resolver:
        async def resolve(request: FastAPIRequest) -> Handler:
            session_id = _session_id(_caller_key(request))
            async with lock:
                entry = sessions.get(session_id)
                if entry is None:
                    client = client_factory(session_id)
                    entry = (client, _collect_handlers(client))
                    sessions[session_id] = entry
                sessions.move_to_end(session_id)
                evicted = []
                while len(sessions) > max_sessions:
                    _, (old, _handlers) = sessions.popitem(last=False)
                    evicted.append(old)
            for old in evicted:
                await _close_client(old)
            return entry[1][path]

        return resolve

    app = FastAPI()
    app.get("/_health")(_health)
    for path in paths:
        app.post(path)(_make_endpoint(path, _session_resolver(path)))
    return app


async def _health() -> dict[str, str]:
    return {"status": "ok"}


def _fixed_resolver(method: Handler) -> Resolver:
    async def resolve(_request: FastAPIRequest) -> Handler:
        return method

    return resolve


def _caller_key(request: FastAPIRequest) -> str:
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _session_id(caller_key: str) -> str:
    if not caller_key:
        return "anonymous"
    # The key is identity, not a secret to echo around: hash it so logs
    # and upstream `x-session-id` headers never carry the raw value.
    return hashlib.sha256(caller_key.encode()).hexdigest()[:24]


async def _close_client(client: Client) -> None:
    if isinstance(client, _AsyncCloseable):
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - eviction must not fail the live request
            logger.exception("abridge serve: failed to close evicted client")


def _make_endpoint(path: str, resolve: Resolver):
    async def endpoint(request: FastAPIRequest) -> Response:
        body = await _read_json(request)
        try:
            handler = await resolve(request)
            response = await handler(Request(path=path, body=body))
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
    parser.add_argument("--host", default="127.0.0.1", help="bind address; expose beyond loopback deliberately")
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument("--upstream-timeout", type=float, default=180.0)
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
            session_id=session_id,
        )

    app = build_session_app(factory, max_sessions=args.max_sessions)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
