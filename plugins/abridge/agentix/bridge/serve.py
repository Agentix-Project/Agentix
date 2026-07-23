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

Two upstream modes, mutually exclusive:

* `--upstream-base-url` — plain OpenAI-compatible engine; translation +
  transport live in `AnthropicFromOpenAIClient` (the openai SDK owns the
  HTTP). The original mode, unchanged.
* `--tito-url` — a token-recording session gateway (the TITO gateway).
  The gateway keeps ONE append-only linear conversation per session,
  while real Anthropic agents multiplex several logical conversations
  over a single API key (helper calls, subagents, a rerun). So each
  caller session DEMUXES by conversation: requests are keyed by
  `sha256(canonical system + first user message)` and each distinct key
  gets its own `AnthropicToOpenAI(SessionForward(tito_url).handler())` —
  its own gateway session. The gateway sees the OpenAI chat body, owns
  render/generate and the token record; this server only adds the
  Anthropic shell, identity stamping, and the demux.

  Demux boundaries (documented, not silently papered over): a history
  rewrite that keeps the first user message (e.g. mid-conversation
  compaction) still lands in the SAME gateway session and is handled by
  the gateway's rollback / from-scratch paths — rewrites deeper than the
  gateway's rollback window are its documented 400. Two genuinely
  different conversations with byte-identical (system, first user
  message) collide into one session. And when the caller-session LRU
  evicts a key (max_sessions), a still-active rollout on that key
  continues in FRESH gateway sessions — the gateway-side capture splits
  at that boundary (turn_index restarts); size `--max-sessions` above
  the number of concurrent rollout keys. `--tito-delete-on-evict` opts
  into reaping the gateway sessions when a caller session closes; the
  default keeps them for harvest.

`--record-dir` (either mode) wraps each session's client in a `Recorder`
writing message-level rows to `<record_dir>/<session_id>.jsonl`; rows
carry `session_id` + `request_id`, and the same `request_id` is stamped
as `x-request-id` on the upstream hop so message rows join token records.

`GET /_health` reports `translation_spec_sha` — the SHA-256 of the
Anthropic<->OpenAI transform module — so downstream data contracts can
pin the exact translation in effect.

    OPENAI_API_KEY=EMPTY agentix-bridge-serve \
        --upstream-base-url http://vllm:8000/v1 --upstream-model qwen3-32b \
        --host 0.0.0.0 --require-key-prefix rollout-secret-
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, Response

from .proxy import (
    AbridgeError,
    Client,
    ClientResponse,
    Handler,
    Request,
    _AsyncCloseable,
    _collect_handlers,
    on,
)

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

    Correlation caveat for `--tito-url` mode: there the upstream
    `x-session-id` is the GATEWAY's own session id (assigned per
    conversation), not this hash — the caller-side↔gateway-side join
    lives in the Recorder rows instead (`session_id` = this hash,
    `gateway_session_id` = the gateway's id, `request_id` = the per-call
    `x-request-id` echoed into the gateway's token records).
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
    body = {"status": "ok"}
    # The translation contract pin: SHA-256 of the transform module source.
    # Downstream data contracts compare it against the value their captured
    # trajectories were produced under. Lazy import: the transforms are
    # SDK-free, but apps serving only custom handlers shouldn't fail health
    # over a broken clients package.
    try:
        from .clients._anthropic_transforms import TRANSLATION_SPEC_SHA

        body["translation_spec_sha"] = TRANSLATION_SPEC_SHA
    except ImportError:  # pragma: no cover - clients package always ships
        pass
    return body


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


def _build_parser() -> argparse.ArgumentParser:
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
        "--tito-url",
        default=os.environ.get("TITO_URL"),
        help=(
            "session-scoped token-recording gateway (the TITO gateway), e.g. "
            "http://tito:30000 — mutually exclusive with --upstream-base-url: the "
            "gateway becomes the upstream, owns render/generate and the token "
            "record, and each caller session maps to one gateway session "
            "(env: TITO_URL)"
        ),
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
    parser.add_argument(
        "--record-dir",
        default=os.environ.get("ABRIDGE_RECORD_DIR"),
        help=(
            "record every served (request, response) pair to "
            "<record-dir>/<session_id>.jsonl via Recorder — message-level rows with "
            "session_id + request_id (+ gateway_session_id in tito mode), flushed "
            "per line (env: ABRIDGE_RECORD_DIR)"
        ),
    )
    parser.add_argument(
        "--tito-delete-on-evict",
        action="store_true",
        help=(
            "tito mode only: DELETE the gateway sessions when a caller session "
            "closes (LRU eviction or shutdown). Default off — gateway sessions "
            "stay alive for harvest, and the harvester deletes them"
        ),
    )
    return parser


def _canonical_text(value: Any) -> str:
    """Flatten Anthropic content (str or block list) to conversation-identity
    text: text blocks contribute their text (cache_control and other
    tokenization-irrelevant decorations are ignored), other blocks their
    sorted-JSON form."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(json.dumps(block, sort_keys=True, ensure_ascii=False, default=repr))
        return "\n".join(parts)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr)


def _conversation_key(body: dict[str, Any]) -> str:
    """A logical-conversation key for an Anthropic Messages request: the
    canonicalized system prompt + first user message. Turns of one
    conversation share it (the head of the history is append-only in normal
    operation); a helper call, subagent, or rerun with a different opening
    gets a different key."""
    first_user = next(
        (m.get("content") for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    blob = _canonical_text(body.get("system")) + "\x00" + _canonical_text(first_user)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class _TitoConversationDemux:
    """One caller key, many logical conversations → one gateway session each.

    The TITO gateway's session is a single append-only linear history, but an
    Anthropic agent multiplexes conversations over one API key (helper calls,
    subagents, reruns). Routing everything into one gateway session bricks
    the key at the first unrelated history (the gateway rightly 400s a
    request sharing no prefix). This client keys each request by
    `_conversation_key` and lazily builds one
    `AnthropicToOpenAI(SessionForward(...).handler())` per conversation —
    scoped like the conversation itself, including the converter's
    assistant-replay memory. The conversation map is bounded by the caller
    session's own lifetime (LRU eviction / shutdown closes all of them).

    `delete_on_close=True` reaps the gateway sessions on `aclose()`
    (`--tito-delete-on-evict`); the default leaves them alive for harvest.
    """

    def __init__(
        self,
        tito_url: str,
        *,
        model: str | None = None,
        timeout: float = 540.0,
        delete_on_close: bool = False,
    ) -> None:
        self._tito_url = tito_url
        self._model = model
        self._timeout = timeout
        self._delete_on_close = delete_on_close
        self._conversations: dict[str, Any] = {}  # key -> AnthropicToOpenAI
        self._forwards: dict[str, Any] = {}  # key -> SessionForward

    def _converter_for(self, body: dict[str, Any]) -> Any:
        key = _conversation_key(body)
        converter = self._conversations.get(key)
        if converter is None:
            from .clients import AnthropicToOpenAI
            from .forward import SessionForward

            forward = SessionForward(self._tito_url, paths=["/v1/chat/completions"], timeout=self._timeout)
            converter = AnthropicToOpenAI(forward.handler(), model=self._model)
            self._conversations[key] = converter
            self._forwards[key] = forward
            logger.info("tito demux: new conversation %s -> fresh gateway session", key)
        return converter

    @on("/v1/messages")
    async def messages(self, request: Request) -> ClientResponse:
        return await self._converter_for(request.body).messages(request)

    @on("/v1/messages/count_tokens")
    async def count_tokens(self, request: Request) -> ClientResponse:
        # Answered locally (same estimate as the converters) — counting must
        # not create a gateway session for a conversation that never runs.
        from .clients._anthropic_transforms import count_anthropic_tokens

        return ClientResponse.json({"input_tokens": count_anthropic_tokens(request.body).input_tokens})

    def environ(self, handle: Any) -> dict[str, str]:
        from .clients.anthropic import PLACEHOLDER_API_KEY

        return {"ANTHROPIC_BASE_URL": handle.url, "ANTHROPIC_API_KEY": PLACEHOLDER_API_KEY}

    async def aclose(self) -> None:
        for key, forward in self._forwards.items():
            if self._delete_on_close:
                try:
                    await forward.delete_session()
                except Exception:  # noqa: BLE001 - best-effort reap
                    logger.exception("tito demux: failed to delete gateway session for conversation %s", key)
        for key, converter in self._conversations.items():
            try:
                await converter.aclose()
            except Exception:  # noqa: BLE001 - close every conversation
                logger.exception("tito demux: failed to close conversation %s", key)
        self._conversations.clear()
        self._forwards.clear()


def _client_factory(args: argparse.Namespace) -> Callable[[str], Client]:
    """The per-caller-session client for the chosen upstream mode, wrapped in
    a `Recorder` when `--record-dir` is set."""
    if args.tito_url:
        # Composition seam (transport-blind): the TITO gateway sees the
        # OpenAI chat body and owns tokens + recording; abridge adds the
        # Anthropic shell and the per-conversation demux (one gateway
        # session per logical conversation on the caller key).
        def build(session_id: str) -> Client:
            return _TitoConversationDemux(
                args.tito_url,
                model=args.upstream_model,
                timeout=args.upstream_timeout,
                delete_on_close=bool(getattr(args, "tito_delete_on_evict", False)),
            )
    else:
        # Lazy: the translation client needs the `openai` extra.
        from .clients import AnthropicFromOpenAIClient

        def build(session_id: str) -> Client:
            return AnthropicFromOpenAIClient(
                base_url=args.upstream_base_url,
                api_key=args.upstream_api_key,
                model=args.upstream_model,
                timeout=args.upstream_timeout,
                max_retries=args.upstream_max_retries,
                session_id=session_id,
            )

    if not args.record_dir:
        return build

    from .recorder import Recorder

    record_dir = Path(args.record_dir)

    def build_recorded(session_id: str) -> Client:
        return Recorder(build(session_id), record_dir / f"{session_id}.jsonl", session_id=session_id)

    return build_recorded


def main(argv: list[str] | None = None) -> None:
    """`agentix-bridge-serve` — an Anthropic-speaking front for an
    OpenAI-compatible engine or a token-recording session gateway, one
    session per caller key."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.upstream_base_url and args.tito_url:
        parser.error("--upstream-base-url and --tito-url are mutually exclusive: the gateway IS the upstream")
    if not args.upstream_base_url and not args.tito_url:
        parser.error("one of --upstream-base-url (env OPENAI_BASE_URL) or --tito-url (env TITO_URL) is required")

    factory = _client_factory(args)

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
