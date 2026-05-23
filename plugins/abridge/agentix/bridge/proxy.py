"""Sandbox-side LLM proxy. Captures Anthropic/OpenAI requests inside
the sandbox and ferries them to the host.

```
agent -> http://127.0.0.1:<port>/v1/messages
              | detect family
              | transform Anthropic -> OpenAI (if needed)
              | capture record
              | SIO request -> /abridge "llm_call"
              v
            host (`OpenAICompatibleClient`)
              | call real OpenAI-compatible upstream
              | return JSON
              v
agent <- (transformed back to Anthropic if needed)
```

Three reasons we run a plain HTTP server (no mitmproxy) inside the
sandbox:

  1. Zero extra system binaries — the previous bridge needed
     `mitmdump` plus a trust-store-aware install. A FastAPI + uvicorn
     setup is already present in the bundle.
  2. The agent harnesses (Anthropic SDK, OpenAI SDK, mini-swe-agent's
     LiteLLM) accept a base URL override (`ANTHROPIC_BASE_URL`,
     `OPENAI_BASE_URL`), so rerouting to `http://127.0.0.1:<port>` is
     a one-env-var move; no TLS interception needed.
  3. The proxy is a single Python module that the host can introspect
     for which requests have flown through it. No subprocess to babysit
     and no mitmproxy hook IPC.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import agentix
from agentix import trace

from .detection import detect
from .storage import CompletionRecord, InMemoryStore, make_record
from .transform import (
    anthropic_messages_to_openai,
    anthropic_sse,
    count_anthropic_tokens,
    openai_to_anthropic_messages,
)

logger = logging.getLogger("agentix.bridge.proxy")
LLM_SPAN_NAME = "llm.request"
"""Span name emitted around every LLM call routed through the proxy.

Each span carries `record_id`, `family`, `request_path`, and (on
success) `model` + `usage.*` attributes; an `error` event is added
when the upstream call fails or times out, along with `set_status`
flipping to `error`. Sub-events fire during translation, capture, and
the host round-trip so consumers can see where time went.
"""

NAMESPACE = "/abridge"
REQUEST_EVENT = "llm_call"
RECORD_EVENT = "completion_record"


# ── sandbox-side: HTTP proxy + SIO emitter ────────────────────────────────


@dataclass(slots=True)
class ProxyHandle:
    """Handle to a running sandbox proxy.

    `anthropic_base_url` and `openai_base_url` are what an agent
    harness should point its SDK at. Both URLs share the same server;
    only the path family differs.
    """

    proxy_id: str
    url: str
    port: int
    anthropic_base_url: str
    openai_base_url: str


@dataclass
class _Running:
    handle: ProxyHandle
    server: uvicorn.Server
    task: asyncio.Task


_running: dict[str, _Running] = {}


class _SandboxNamespace(agentix.Namespace):
    """SIO namespace the sandbox proxy uses to talk to the host.

    Outbound events: `llm_call` (round-trip), `completion_record`
    (fire-and-forget telemetry).
    """

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


async def start_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout: float = 600.0,
) -> ProxyHandle:
    """Start the sandbox-side HTTP proxy.

    Returns once the server is bound. Idempotent only per-call: each
    invocation starts a fresh proxy with a new id, so call `stop_proxy`
    when done. (For repeated agents in one sandbox, share the handle.)
    """
    ns = _get_namespace()
    app = _build_app(ns=ns, request_timeout=request_timeout)
    bound_port = port or _free_port(host)
    config = uvicorn.Config(app, host=host, port=bound_port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await _wait_uvicorn_started(server)

    proxy_id = uuid.uuid4().hex
    url = f"http://{host}:{bound_port}"
    handle = ProxyHandle(
        proxy_id=proxy_id,
        url=url,
        port=bound_port,
        anthropic_base_url=url,
        openai_base_url=f"{url}/v1",
    )
    _running[proxy_id] = _Running(handle=handle, server=server, task=task)
    logger.info("abridge proxy %s listening on %s", proxy_id, url)
    return handle


async def stop_proxy(handle: ProxyHandle) -> None:
    rec = _running.pop(handle.proxy_id, None)
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


def _build_app(*, ns: _SandboxNamespace, request_timeout: float) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/_health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await _handle_request(
            ns=ns,
            request=request,
            api_path="/v1/messages",
            request_timeout=request_timeout,
        )

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request) -> Response:
        body = await _read_json(request)
        result = count_anthropic_tokens(body)
        return JSONResponse({"input_tokens": result.input_tokens})

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        return await _handle_request(
            ns=ns,
            request=request,
            api_path="/v1/chat/completions",
            request_timeout=request_timeout,
        )

    return app


async def _read_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        return _json_loads(raw)
    except ValueError:
        return {}


def _json_loads(blob: bytes) -> dict[str, Any]:
    import json as _json

    parsed = _json.loads(blob)
    if not isinstance(parsed, dict):
        raise ValueError("LLM request bodies must be JSON objects")
    return parsed


async def _handle_request(
    *,
    ns: _SandboxNamespace,
    request: Request,
    api_path: str,
    request_timeout: float,
) -> Response:
    body = await _read_json(request)
    family = detect(api_path, body)
    record_id = uuid.uuid4().hex
    started_at = time.time()
    client_stream = bool(body.get("stream"))
    requested_model = str(body.get("model") or "")

    with trace.span(
        LLM_SPAN_NAME,
        record_id=record_id,
        family=family.value,
        request_path=api_path,
        model=requested_model,
        stream=client_stream,
    ) as llm_span:
        logger.info(
            "abridge.llm.request",
            extra={
                "record_id": record_id,
                "family": family.value,
                "request_path": api_path,
                "model": requested_model,
            },
        )
        llm_span.add_event("capture.request", model=requested_model)

        if family.is_anthropic:
            llm_span.add_event("translate.start", direction="anthropic->openai")
            logger.debug(
                "abridge.llm.translate.request",
                extra={"record_id": record_id, "direction": "anthropic->openai"},
            )
            upstream_body = anthropic_messages_to_openai(body)
            llm_span.add_event("translate.end", direction="anthropic->openai")
        else:
            upstream_body = dict(body)
        upstream_body["stream"] = False

        payload = {
            "record_id": record_id,
            "family": family.value,
            "request_path": api_path,
            "upstream_body": upstream_body,
            "client_stream": client_stream,
            "anthropic_model": requested_model if family.is_anthropic else None,
        }

        llm_span.add_event("upstream.start", model=upstream_body.get("model"))
        logger.info(
            "abridge.llm.upstream.start",
            extra={"record_id": record_id, "upstream_model": upstream_body.get("model")},
        )
        try:
            result = await asyncio.wait_for(
                ns.request(REQUEST_EVENT, payload), timeout=request_timeout
            )
        except TimeoutError:
            err_message = "proxy timed out waiting for host"
            logger.error(
                "abridge.llm.timeout",
                extra={"record_id": record_id, "timeout_s": request_timeout},
            )
            llm_span.add_event(
                "upstream.error", error="timeout", timeout_s=request_timeout
            )
            llm_span.set_error(err_message, kind="timeout")
            record = make_record(
                request_id=record_id,
                family=family,
                started_at=started_at,
                request_path=api_path,
                request_body=body,
                upstream_body=upstream_body,
                response_body=None,
                status="timeout",
                error=err_message,
            )
            await _emit_record(ns, record)
            return JSONResponse(
                {"error": {"type": "timeout", "message": err_message}}, status_code=504
            )

        if not isinstance(result, dict):
            raise RuntimeError(f"abridge host returned non-dict result: {result!r}")
        if "error" in result:
            err = result["error"]
            message = str(err.get("message") if isinstance(err, dict) else err)
            status = int(err.get("status_code", 502)) if isinstance(err, dict) else 502
            llm_span.add_event("upstream.error", error=message, status_code=status)
            llm_span.set_error(message, kind="upstream", status_code=status)
            logger.warning(
                "abridge.llm.upstream.error",
                extra={
                    "record_id": record_id,
                    "status_code": status,
                    "message": message[:200],
                },
            )
            record = make_record(
                request_id=record_id,
                family=family,
                started_at=started_at,
                request_path=api_path,
                request_body=body,
                upstream_body=upstream_body,
                response_body=None,
                status="error",
                error=message,
            )
            await _emit_record(ns, record)
            return JSONResponse(
                {"error": {"type": "upstream_error", "message": message}}, status_code=status
            )

        openai_response = result.get("response") or {}
        if not isinstance(openai_response, dict):
            raise RuntimeError("abridge host returned non-dict response")
        llm_span.add_event(
            "upstream.end",
            response_id=str(openai_response.get("id", "")),
            finish_reason=str(
                ((openai_response.get("choices") or [{}])[0] or {}).get("finish_reason", "")
            ),
        )

        if family.is_anthropic:
            llm_span.add_event("translate.start", direction="openai->anthropic")
            anthropic_body = openai_to_anthropic_messages(
                openai_response, response_model=requested_model
            )
            llm_span.add_event("translate.end", direction="openai->anthropic")
            response_body: dict[str, Any] = anthropic_body
            if client_stream:
                envelope = anthropic_sse(anthropic_body)
                response: Response = Response(content=envelope, media_type="text/event-stream")
            else:
                response = JSONResponse(anthropic_body)
        else:
            response_body = openai_response
            response = JSONResponse(openai_response)

        record = make_record(
            request_id=record_id,
            family=family,
            started_at=started_at,
            request_path=api_path,
            request_body=body,
            upstream_body=upstream_body,
            response_body=response_body,
        )
        llm_span.attrs.update(
            {
                "usage.prompt_tokens": record.usage.prompt_tokens,
                "usage.completion_tokens": record.usage.completion_tokens,
                "usage.cached_tokens": record.usage.cached_tokens,
                "usage.reasoning_tokens": record.usage.reasoning_tokens,
                "usage.total_tokens": record.usage.total_tokens,
                "duration_ms": record.duration_ms,
            }
        )
        llm_span.set_status("ok")
        logger.info(
            "abridge.llm.response",
            extra={
                "record_id": record_id,
                "prompt_tokens": record.usage.prompt_tokens,
                "completion_tokens": record.usage.completion_tokens,
                "duration_ms": record.duration_ms,
            },
        )
        await _emit_record(ns, record)
        return response


async def _emit_record(ns: _SandboxNamespace, record: CompletionRecord) -> None:
    """Fire-and-forget capture record to the host. Logging is best-effort."""
    try:
        await ns.emit(RECORD_EVENT, record.to_dict())
    except Exception:
        logger.exception("abridge: completion_record emit failed")


# ── env helpers ───────────────────────────────────────────────────────────


def export_environ(handle: ProxyHandle) -> dict[str, str]:
    """Env vars an agent harness should inherit to route through the proxy.

    Pre-fills the standard base-URL overrides for the Anthropic SDK,
    OpenAI SDK, and `LiteLLM`. The proxy itself does not validate API
    keys; the host's `OpenAICompatibleClient` supplies the real
    upstream credentials.
    """
    return {
        "ANTHROPIC_BASE_URL": handle.anthropic_base_url,
        "OPENAI_BASE_URL": handle.openai_base_url,
        "OPENAI_API_BASE": handle.openai_base_url,
        "ABRIDGE_PROXY_URL": handle.url,
        # Many SDKs refuse to ride a missing key entirely; let the agent
        # supply a placeholder. The host substitutes the real key.
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", "sk-abridge"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-abridge"),
    }


__all__ = [
    "InMemoryStore",
    "NAMESPACE",
    "ProxyHandle",
    "RECORD_EVENT",
    "REQUEST_EVENT",
    "export_environ",
    "start_proxy",
    "stop_proxy",
]
