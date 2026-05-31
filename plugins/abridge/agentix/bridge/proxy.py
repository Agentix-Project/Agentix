"""Sandbox-side LLM proxy. Captures Anthropic/OpenAI requests inside
the sandbox and ferries them to the host.

```
agent -> http://127.0.0.1:<port>/v1/messages
              | detect family
              | transform Anthropic -> OpenAI (if needed)
              | capture record
              | SIO request -> /abridge "llm_call"
              v
            host (`Bridge`)
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
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import agentix
from agentix.utils import trace

from .detection import ApiFamily, detect
from .storage import CompletionRecord, InMemoryStore, make_record
from .transform import (
    anthropic_messages_to_openai,
    anthropic_sse,
    count_anthropic_tokens,
    openai_to_anthropic_messages,
)

logger = logging.getLogger("agentix.bridge.proxy")

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
    session_id: str | None = None,
    parent_trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> ProxyHandle:
    """Start the sandbox-side HTTP proxy.

    Returns once the server is bound. Idempotent only per-call: each
    invocation starts a fresh proxy with a new id, so call `stop_proxy`
    when done. (For repeated agents in one sandbox, share the handle.)

    `session_id`, when set, is stamped onto every captured call so the
    host/gateway can group a rollout's LLM calls into one trajectory.
    `Bridge.start_proxy(...)` wires this for you; the caller (host) supplies
    the id so it can later query the gateway by the same key.
    """
    ns = _get_namespace()
    app = _build_app(
        ns=ns,
        request_timeout=request_timeout,
        session_id=session_id,
        parent_trace_id=parent_trace_id,
        parent_span_id=parent_span_id,
    )
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


def _build_app(
    *,
    ns: _SandboxNamespace,
    request_timeout: float,
    session_id: str | None = None,
    parent_trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> FastAPI:
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
            session_id=session_id,
            parent_trace_id=parent_trace_id,
            parent_span_id=parent_span_id,
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
            session_id=session_id,
            parent_trace_id=parent_trace_id,
            parent_span_id=parent_span_id,
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
    session_id: str | None = None,
    parent_trace_id: str | None = None,
    parent_span_id: str | None = None,
) -> Response:
    body = await _read_json(request)
    family = detect(api_path, body)
    record_id = uuid.uuid4().hex
    started_at = time.time()
    # Synthetic carrier for the host's rollout span, so this LLM span nests
    # under it (shared trace_id + parent_id) — one trace per rollout.
    parent_span = (
        trace.Span(span_id=parent_span_id, trace_id=parent_trace_id, parent_id=None, name="rollout")
        if parent_span_id and parent_trace_id
        else None
    )

    if family.is_anthropic:
        upstream_body = anthropic_messages_to_openai(body)
    else:
        upstream_body = dict(body)
    upstream_body["stream"] = False

    payload = {
        "record_id": record_id,
        "session_id": session_id,
        "family": family.value,
        "request_path": api_path,
        "upstream_body": upstream_body,
        "client_stream": bool(body.get("stream")),
        "anthropic_model": str(body.get("model") or "") if family.is_anthropic else None,
    }

    # One span per LLM call, opened around the in-flight request so its
    # timing reflects what the agent observed (incl. the SIO round-trip).
    # Ships over `/trace`; rollout calls correlate by `agentix.session_id`
    # (cross-HTTP span parenting isn't available without the agent
    # propagating traceparent, and the agent never intervenes).
    with trace.span(
        f"chat {body.get('model') or 'unknown'}",
        parent=parent_span,
        **_llm_request_attrs(family, body, session_id=session_id, record_id=record_id),
    ) as sp:
        try:
            result = await asyncio.wait_for(ns.request(REQUEST_EVENT, payload), timeout=request_timeout)
        except TimeoutError:
            sp.set_error("proxy timed out waiting for host")
            record = make_record(
                request_id=record_id,
                session_id=session_id,
                family=family,
                started_at=started_at,
                request_path=api_path,
                request_body=body,
                upstream_body=upstream_body,
                response_body=None,
                status="timeout",
                error="proxy timed out waiting for host",
            )
            await _emit_record(ns, record)
            return JSONResponse(
                {"error": {"type": "timeout", "message": record.error}}, status_code=504
            )

        if not isinstance(result, dict):
            raise RuntimeError(f"abridge host returned non-dict result: {result!r}")
        if "error" in result:
            err = result["error"]
            message = str(err.get("message") if isinstance(err, dict) else err)
            sp.set_error(message)
            record = make_record(
                request_id=record_id,
                session_id=session_id,
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
            status = int(err.get("status_code", 502)) if isinstance(err, dict) else 502
            return JSONResponse({"error": {"type": "upstream_error", "message": message}}, status_code=status)

        openai_response = result.get("response") or {}
        if not isinstance(openai_response, dict):
            raise RuntimeError("abridge host returned non-dict response")

        if family.is_anthropic:
            anthropic_body = openai_to_anthropic_messages(
                openai_response, response_model=str(body.get("model") or "")
            )
            response_body: dict[str, Any] = anthropic_body
            if body.get("stream"):
                envelope = anthropic_sse(anthropic_body)
                response: Response = Response(content=envelope, media_type="text/event-stream")
            else:
                response = JSONResponse(anthropic_body)
        else:
            response_body = openai_response
            response = JSONResponse(openai_response)

        record = make_record(
            request_id=record_id,
            session_id=session_id,
            family=family,
            started_at=started_at,
            request_path=api_path,
            request_body=body,
            upstream_body=upstream_body,
            response_body=response_body,
        )
        _apply_response_span(sp, record)
        await _emit_record(ns, record)
        return response


# ── OTel GenAI span attributes (abridge is a span producer; /trace owns export) ──


def _llm_request_attrs(
    family: ApiFamily, body: dict[str, Any], *, session_id: str | None, record_id: str
) -> dict[str, Any]:
    """Initial span attributes from the request, per OTel GenAI conventions."""
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "anthropic" if family.is_anthropic else "openai",
        "gen_ai.request.model": str(body.get("model") or ""),
        "agentix.request_id": record_id,
    }
    if session_id:
        attrs["agentix.session_id"] = session_id
    for key, attr in (
        ("max_tokens", "gen_ai.request.max_tokens"),
        ("temperature", "gen_ai.request.temperature"),
        ("top_p", "gen_ai.request.top_p"),
    ):
        if body.get(key) is not None:
            attrs[attr] = body[key]

    # Prompt content (system + messages) so backends show the input, not just
    # tokens. GenAI `gen_ai.prompt.{i}.role/content` is what LangSmith/Langfuse read.
    idx = 0
    system = body.get("system")
    if system:
        attrs["gen_ai.prompt.0.role"] = "system"
        attrs["gen_ai.prompt.0.content"] = _content_to_text(system)
        idx = 1
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        attrs[f"gen_ai.prompt.{idx}.role"] = str(msg.get("role", "user"))
        attrs[f"gen_ai.prompt.{idx}.content"] = _content_to_text(msg.get("content", ""))
        idx += 1
    return attrs


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic/OpenAI message content (str or block list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                bt = block.get("type")
                if bt == "text":
                    parts.append(str(block.get("text", "")))
                elif bt == "tool_use":
                    parts.append(f"[tool_use {block.get('name', '')} {json.dumps(block.get('input') or {})}]")
                elif bt == "tool_result":
                    parts.append(f"[tool_result] {_content_to_text(block.get('content'))}")
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def _response_content(rb: dict[str, Any], *, family: ApiFamily) -> Any:
    if family.is_anthropic:
        return rb.get("content")
    choice = (rb.get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content") if isinstance(choice, dict) else None


def _apply_response_span(sp: trace.Span, record: CompletionRecord) -> None:
    """Backfill usage / response model / tool calls once the reply lands."""
    sp.set_attributes(
        **{
            "gen_ai.usage.input_tokens": record.usage.prompt_tokens,
            "gen_ai.usage.output_tokens": record.usage.completion_tokens,
        }
    )
    rb = record.response_body
    if not isinstance(rb, dict):
        return
    model = rb.get("model")
    if isinstance(model, str) and model:
        sp.set_attribute("gen_ai.response.model", model)
    # Completion content so the backend shows the output, not just tokens.
    sp.set_attribute("gen_ai.completion.0.role", "assistant")
    sp.set_attribute("gen_ai.completion.0.content", _content_to_text(_response_content(rb, family=record.family)))
    names = _tool_call_names(rb, family=record.family)
    if names:
        # The model's tool requests, derived from the response — no agent
        # instrumentation. Full tool/observation span stitching (across
        # calls) is a follow-up; it intersects the compaction / sub-agent
        # open problems.
        sp.add_event("gen_ai.tool_calls", names=names, count=len(names))


def _tool_call_names(response_body: dict[str, Any], *, family: ApiFamily) -> list[str]:
    names: list[str] = []
    if family.is_anthropic:
        for block in response_body.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name")
                if isinstance(name, str):
                    names.append(name)
        return names
    for choice in response_body.get("choices") or []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        for tc in (msg or {}).get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = (fn or {}).get("name")
            if isinstance(name, str):
                names.append(name)
    return names


async def _emit_record(ns: _SandboxNamespace, record: CompletionRecord) -> None:
    """Fire-and-forget capture record to the host. Logging is best-effort."""
    try:
        await ns.emit(RECORD_EVENT, record.to_dict())
    except Exception:
        logger.exception("abridge: completion_record emit failed")


# ── env helpers ───────────────────────────────────────────────────────────


_PLACEHOLDER_KEY = "sk-abridge"


def export_environ(handle: ProxyHandle) -> dict[str, str]:
    """Env vars an agent harness should inherit to route through the proxy.

    Pure function of the handle — safe to call host-side. Pre-fills the
    standard base-URL overrides for the Anthropic SDK, OpenAI SDK, and
    `LiteLLM`. The API-key values are a constant placeholder: SDKs refuse
    to start without *some* key, but the real upstream credentials live
    only on the host (the gateway substitutes them), so the placeholder
    must never be a real key — reading one from the environment here would
    leak it into the sandbox.
    """
    return {
        "ANTHROPIC_BASE_URL": handle.anthropic_base_url,
        "OPENAI_BASE_URL": handle.openai_base_url,
        "OPENAI_API_BASE": handle.openai_base_url,
        "ABRIDGE_PROXY_URL": handle.url,
        "ANTHROPIC_API_KEY": _PLACEHOLDER_KEY,
        "OPENAI_API_KEY": _PLACEHOLDER_KEY,
    }


__all__ = [
    "NAMESPACE",
    "InMemoryStore",
    "ProxyHandle",
    "RECORD_EVENT",
    "REQUEST_EVENT",
    "export_environ",
    "start_proxy",
    "stop_proxy",
]
