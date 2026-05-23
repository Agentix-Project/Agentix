"""mitmproxy-backed Abridge prototype.

Sandbox flow:

    agent process -> sandbox mitmproxy -> sandbox forwarder
      -> Agentix SIO -> host OpenAIForwarder -> upstream

The mitmproxy addon and forwarder run inside the sandbox. The real
upstream API key and OpenAI-compatible request live on the host.
"""

import asyncio
import base64
import contextlib
import importlib
import inspect
import json
import logging
import os
import socket
import sys
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import agentix
from agentix import AsyncClientNamespace

logger = logging.getLogger("agentix.bridge.mitm")
NAMESPACE = "/abridge-mitm"
_MITM_HTTP: Any | None = None
HookHandler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any] | None] | None]


@dataclass
class ProxyHandle:
    id: str
    url: str
    port: int
    forwarder_url: str
    forwarder_port: int
    mode: str


@dataclass
class _RunningProxy:
    handle: ProxyHandle
    forwarder_server: uvicorn.Server
    forwarder_task: asyncio.Task
    process: asyncio.subprocess.Process
    log_task: asyncio.Task


class _SandboxNamespace(agentix.Namespace):
    namespace = NAMESPACE


_namespace_singleton: _SandboxNamespace | None = None
_running: dict[str, _RunningProxy] = {}


def _get_namespace() -> _SandboxNamespace:
    global _namespace_singleton
    if _namespace_singleton is None:
        _namespace_singleton = _SandboxNamespace()
        agentix.register_namespace(_namespace_singleton)
    return _namespace_singleton


async def start_proxy(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    mode: str = "reverse:https://api.anthropic.com",
    forwarder_host: str = "127.0.0.1",
    forwarder_port: int = 0,
    request_timeout: float = 600.0,
    extra_mitm_args: list[str] | None = None,
) -> ProxyHandle:
    """Start sandbox-local mitmproxy plus a local SIO forwarder.

    Return `handle.url` to sandbox agents as their Anthropic base URL.
    In the default reverse mode, no Docker-specific host route is needed:
    the agent only talks to 127.0.0.1 inside its own sandbox.
    """
    ns = _get_namespace()
    app = _build_forwarder_app(ns=ns, request_timeout=request_timeout)
    f_config = uvicorn.Config(app, host=forwarder_host, port=forwarder_port, log_level="warning")
    f_server = uvicorn.Server(f_config)
    f_task = asyncio.create_task(f_server.serve())
    bound_forwarder_port = await _wait_uvicorn_started(f_server, forwarder_port)
    forwarder_url = f"http://{forwarder_host}:{bound_forwarder_port}"

    bound_proxy_port = port or _free_tcp_port(host)
    args = [
        "--mode",
        mode,
        "--listen-host",
        host,
        "--listen-port",
        str(bound_proxy_port),
        "--set",
        "block_global=false",
        "--quiet",
    ]
    if extra_mitm_args:
        args.extend(extra_mitm_args)

    env = dict(os.environ)
    env["ABRIDGE_HOOK_URL"] = _join_url_path(forwarder_url, "/hook")
    env.setdefault("ABRIDGE_TRACE", "1")
    env.setdefault("ABRIDGE_ANTHROPIC_HOSTS", "api.anthropic.com,localhost,127.0.0.1")

    code = f"from agentix.bridge.mitm import main; raise SystemExit(main({args!r}))"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    log_task = asyncio.create_task(_drain_mitm_logs(process.stdout))
    try:
        await _wait_port(host, bound_proxy_port, process)
    except Exception:
        process.terminate()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
        f_server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(f_task, timeout=5)
        raise

    sid = uuid.uuid4().hex
    handle = ProxyHandle(
        id=sid,
        url=f"http://{host}:{bound_proxy_port}",
        port=bound_proxy_port,
        forwarder_url=forwarder_url,
        forwarder_port=bound_forwarder_port,
        mode=mode,
    )
    _running[sid] = _RunningProxy(
        handle=handle,
        forwarder_server=f_server,
        forwarder_task=f_task,
        process=process,
        log_task=log_task,
    )
    return handle


async def stop_proxy(handle: ProxyHandle) -> None:
    rec = _running.pop(handle.id, None)
    if rec is None:
        return
    rec.process.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(rec.process.wait(), timeout=5)
    if rec.process.returncode is None:
        rec.process.kill()
        await rec.process.wait()
    rec.log_task.cancel()
    await asyncio.gather(rec.log_task, return_exceptions=True)

    rec.forwarder_server.should_exit = True
    with contextlib.suppress(Exception):
        await asyncio.wait_for(rec.forwarder_task, timeout=5)


class HookForwarder(AsyncClientNamespace):
    """Host-side receiver for protocol-neutral mitmproxy hook events."""

    def __init__(self, handler: HookHandler | None = None) -> None:
        super().__init__(NAMESPACE)
        self._handler = handler

    async def on_proxy_event(self, payload: dict[str, Any]) -> None:
        req_id = payload.get("request_id")
        event = payload.get("data") or {}
        if not isinstance(req_id, str):
            logger.warning("abridge mitm: dropped proxy_event with no request_id")
            return
        if not isinstance(event, dict):
            event = {"kind": "bad_event", "raw": event}
        try:
            value = await self.handle_event(event)
        except Exception as exc:
            logger.exception("abridge mitm: proxy_event failed")
            await self.emit(
                "proxy_event:error",
                {
                    "request_id": req_id,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            return
        await self.emit("proxy_event:result", {"request_id": req_id, "value": value or _continue_action()})

    async def handle_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if self._handler is None:
            return _continue_action()
        result = self._handler(event)
        if inspect.isawaitable(result):
            return await result
        return result


class OpenAIForwarder(HookForwarder):
    """Host-side SIO handler for sandbox mitmproxy flows."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._extra_body = extra_body or {}
        self._timeout = timeout

    async def handle_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if event.get("kind") != "http_request":
            return _continue_action()
        request = event.get("request") or {}
        if not isinstance(request, dict):
            return _continue_action()
        path = str(request.get("path") or "").split("?", 1)[0]
        if path not in {"/v1/messages", "/v1/messages/count_tokens"}:
            return _continue_action()
        return _respond_action(await self.forward(event))

    async def forward(self, event: dict[str, Any]) -> dict[str, Any]:
        request = event.get("request") or {}
        if not isinstance(request, dict):
            return _response_envelope(400, b"bad forwarded request")

        path = str(request.get("path") or "").split("?", 1)[0]
        body = request.get("json") or {}
        if not isinstance(body, dict):
            body = {}

        if path == "/v1/messages/count_tokens":
            return _json_envelope(_count_tokens(body))
        if path != "/v1/messages":
            return _response_envelope(404, f"unsupported captured path: {path}".encode())

        client_stream = bool(body.get("stream"))
        response_model = str(body.get("model") or "")
        openai_body = _anthropic_to_openai_body(
            body,
            upstream_model=self._model,
            extra_body=self._extra_body,
        )
        openai_body["stream"] = False
        upstream = await self._post_openai(openai_body)
        anthropic_body = _openai_to_anthropic_body(upstream, response_model=response_model)
        if client_stream:
            return _response_envelope(200, _anthropic_sse(anthropic_body), content_type="text/event-stream")
        return _json_envelope(anthropic_body)

    async def _post_openai(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
            "accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                _openai_chat_completions_url(self._base_url),
                headers=headers,
                json=body,
            )
        resp.raise_for_status()
        value = resp.json()
        if not isinstance(value, dict):
            raise ValueError("openai-compatible upstream returned non-object JSON")
        return value


def _build_forwarder_app(*, ns: _SandboxNamespace, request_timeout: float) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "abridge-mitm-forwarder"}

    @app.post("/hook")
    async def hook(request: Request) -> JSONResponse:
        event = await request.json()
        try:
            value = await ns.request("proxy_event", event, timeout=request_timeout)
        except agentix.RemoteSioError as exc:
            value = _error_action(
                {"error": {"type": exc.type, "message": exc.message}},
                status_code=502,
            )
        except Exception as exc:
            logger.exception("abridge mitm forwarder failed")
            value = _error_action(
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
                status_code=500,
            )
        if not isinstance(value, dict):
            value = _error_action(
                {"error": {"type": "BadProxyResponse", "message": "host returned non-object response"}},
                status_code=502,
            )
        return JSONResponse(content=value)

    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Run mitmdump with this file loaded as the addon script."""
    try:
        mitm_main = importlib.import_module("mitmproxy.tools.main")
    except ModuleNotFoundError as exc:
        if exc.name == "mitmproxy":
            print(
                "mitmproxy is required for abridge-mitm. "
                "Install the mitm extra or run with `uv run --extra mitm abridge-mitm`.",
                file=sys.stderr,
            )
            return 2
        raise

    mitmdump = getattr(mitm_main, "mitmdump")
    mitmdump(_mitmdump_args(argv))
    return 0


def request(flow: Any) -> None:
    """mitmproxy HTTP request hook."""
    host = flow.request.pretty_host
    path = flow.request.path
    _trace("http.request", method=flow.request.method, host=host, path=path)

    if _is_blocked_host(host):
        _trace("http.block", host=host, path=path)
        flow.kill()
        return

    if _is_anthropic_health(flow):
        _set_response(flow, 200, b"")
        return

    if _is_anthropic_count_tokens(flow):
        body = _json_request_body(flow)
        _apply_response_envelope(flow, _json_envelope(_count_tokens(body)))
        return

    if not _is_anthropic_messages(flow):
        return

    body = _json_request_body(flow)
    hook_url = _hook_url()
    if hook_url:
        try:
            action = _send_hook_event(_http_request_event(flow, body), hook_url=hook_url)
            if _apply_http_action(flow, action):
                return
        except Exception as exc:
            _trace("proxy_event.error", kind="http_request", error=repr(exc))
            _set_response(flow, 502, f"abridge hook failed: {exc}".encode())
        return

    client_stream = bool(body.get("stream"))
    response_model = str(body.get("model") or "")
    openai_body = _anthropic_to_openai_body(body)
    openai_body["stream"] = False

    _trace(
        "anthropic.to_openai.host_request",
        target=_openai_chat_completions_url(),
        model=openai_body["model"],
        client_stream=client_stream,
    )
    try:
        openai_body = _call_openai_compatible(openai_body)
    except Exception as exc:
        _trace("anthropic.to_openai.host_error", error=repr(exc))
        _set_response(flow, 502, f"openai-compatible upstream failed: {exc}".encode())
        return

    anthropic_body = _openai_to_anthropic_body(openai_body, response_model=response_model)
    if client_stream:
        content = _anthropic_sse(anthropic_body)
        content_type = "text/event-stream"
    else:
        content = json.dumps(anthropic_body, separators=(",", ":")).encode()
        content_type = "application/json"
    _set_response(flow, 200, content, content_type=content_type)
    _trace("anthropic.to_openai.host_response", client_stream=client_stream, bytes=len(content))


def response(flow: Any) -> None:
    """mitmproxy HTTP response hook."""
    if flow.response is None:
        return
    _trace(
        "http.response",
        host=flow.request.pretty_host,
        path=flow.request.path,
        status_code=flow.response.status_code,
    )
    _send_hook_event_if_enabled(_http_response_event(flow))


def websocket_message(flow: Any) -> None:
    msg = flow.websocket.messages[-1]
    _trace(
        "websocket.message",
        host=flow.request.pretty_host,
        path=flow.request.path,
        bytes=len(msg.content),
    )
    _send_hook_event_if_enabled(_websocket_message_event(flow, msg))


def tcp_message(flow: Any) -> None:
    msg = flow.messages[-1]
    _trace(
        "tcp.message",
        client=str(getattr(flow.client_conn, "address", "")),
        server=str(getattr(flow.server_conn, "address", "")),
        bytes=len(msg.content),
    )
    _send_hook_event_if_enabled(_tcp_message_event(flow, msg))


def udp_message(flow: Any) -> None:
    msg = flow.messages[-1]
    _trace(
        "udp.message",
        client=str(getattr(flow.client_conn, "address", "")),
        server=str(getattr(flow.server_conn, "address", "")),
        bytes=len(msg.content),
    )
    _send_hook_event_if_enabled(_udp_message_event(flow, msg))


def dns_request(flow: Any) -> None:
    questions = getattr(flow.request, "questions", None) or []
    names = [str(getattr(q, "name", "")) for q in questions]
    _trace("dns.request", names=names)
    _send_hook_event_if_enabled(_dns_request_event(flow, names))


def _is_anthropic_messages(flow: Any) -> bool:
    hosts = _csv_env("ABRIDGE_ANTHROPIC_HOSTS", "api.anthropic.com")
    return (
        flow.request.method.upper() == "POST"
        and _host_matches(flow.request.pretty_host, hosts)
        and flow.request.path.split("?", 1)[0] == "/v1/messages"
    )


def _is_anthropic_health(flow: Any) -> bool:
    hosts = _csv_env("ABRIDGE_ANTHROPIC_HOSTS", "api.anthropic.com")
    return _host_matches(flow.request.pretty_host, hosts) and flow.request.path.split("?", 1)[0] == "/"


def _is_anthropic_count_tokens(flow: Any) -> bool:
    hosts = _csv_env("ABRIDGE_ANTHROPIC_HOSTS", "api.anthropic.com")
    return (
        flow.request.method.upper() == "POST"
        and _host_matches(flow.request.pretty_host, hosts)
        and flow.request.path.split("?", 1)[0] == "/v1/messages/count_tokens"
    )


def _is_blocked_host(host: str) -> bool:
    patterns = _csv_env("ABRIDGE_BLOCK_HOSTS", "segment.io,telemetry")
    return any(pattern and pattern in host for pattern in patterns)


def _anthropic_to_openai_body(
    body: dict[str, Any],
    *,
    upstream_model: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "\n".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if text:
            messages.append({"role": "system", "content": text})

    messages.extend(_anthropic_messages_to_openai(body.get("messages") or []))

    out: dict[str, Any] = {
        "model": upstream_model or os.getenv("OPENAI_MODEL") or os.getenv("ABRIDGE_OPENAI_MODEL") or body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
    }
    for key in ("temperature", "top_p", "stop"):
        if key in body:
            out[key] = body[key]

    tools = _anthropic_tools_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools

    env_extra_body = os.getenv("ABRIDGE_OPENAI_EXTRA_BODY")
    if env_extra_body:
        extra = json.loads(env_extra_body)
        if not isinstance(extra, dict):
            raise ValueError("ABRIDGE_OPENAI_EXTRA_BODY must decode to a JSON object")
        out.update(extra)
    if extra_body:
        out.update(extra_body)

    return out


def _anthropic_messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue

        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                out.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input") or {}),
                                },
                            }
                        ],
                    }
                )
            elif block_type == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _tool_result_text(block.get("content")),
                    }
                )
        if text_parts:
            out.append({"role": role, "content": "\n".join(text_parts)})
    return out


def _anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or "name" not in tool:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
        )
    return out


def _openai_to_anthropic_body(openai_body: dict[str, Any], *, response_model: str) -> dict[str, Any]:
    choice = (openai_body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = openai_body.get("usage") or {}
    content: list[dict[str, Any]] = []

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name", ""),
                "input": args,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    finish_reason = choice.get("finish_reason")
    stop_reason = "end_turn"
    if finish_reason == "tool_calls" or any(block["type"] == "tool_use" for block in content):
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": response_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _anthropic_sse(body: dict[str, Any]) -> bytes:
    content = body.get("content") or []
    usage = body.get("usage") or {}
    parts = [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": body["id"],
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", ""),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
                },
            },
        )
    ]

    for index, block in enumerate(content):
        block_type = block.get("type")
        if block_type == "text":
            parts.append(
                _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            text = block.get("text", "")
            if text:
                parts.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                )
            parts.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        elif block_type == "tool_use":
            parts.append(
                _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "input": {},
                        },
                    },
                )
            )
            partial_json = json.dumps(block.get("input") or {}, separators=(",", ":"))
            parts.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": partial_json},
                    },
                )
            )
            parts.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))

    parts.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": body.get("stop_reason", "end_turn"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    parts.append(_sse("message_stop", {"type": "message_stop"}))
    return b"".join(parts)


def _http_request_event(flow: Any, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "http_request",
        "protocol": "http",
        "hook": "request",
        "flow_id": str(getattr(flow, "id", "")),
        "request": {
            "method": flow.request.method,
            "scheme": flow.request.scheme,
            "host": flow.request.pretty_host,
            "port": flow.request.port,
            "path": flow.request.path,
            "headers": dict(flow.request.headers),
            "json": body,
            **_content_payload(getattr(flow.request, "content", b"")),
        },
    }


def _http_response_event(flow: Any) -> dict[str, Any]:
    return {
        "kind": "http_response",
        "protocol": "http",
        "hook": "response",
        "flow_id": str(getattr(flow, "id", "")),
        "request": {
            "method": flow.request.method,
            "scheme": flow.request.scheme,
            "host": flow.request.pretty_host,
            "port": flow.request.port,
            "path": flow.request.path,
            "headers": dict(flow.request.headers),
        },
        "response": {
            "status_code": flow.response.status_code,
            "headers": dict(flow.response.headers),
            **_content_payload(getattr(flow.response, "content", b"")),
        },
    }


def _websocket_message_event(flow: Any, msg: Any) -> dict[str, Any]:
    return {
        "kind": "websocket_message",
        "protocol": "websocket",
        "hook": "message",
        "flow_id": str(getattr(flow, "id", "")),
        "request": {
            "host": flow.request.pretty_host,
            "path": flow.request.path,
            "headers": dict(flow.request.headers),
        },
        "message": {
            "from_client": bool(getattr(msg, "from_client", False)),
            **_content_payload(getattr(msg, "content", b"")),
        },
    }


def _tcp_message_event(flow: Any, msg: Any) -> dict[str, Any]:
    return {
        "kind": "tcp_message",
        "protocol": "tcp",
        "hook": "message",
        "flow_id": str(getattr(flow, "id", "")),
        "client": str(getattr(flow.client_conn, "address", "")),
        "server": str(getattr(flow.server_conn, "address", "")),
        "message": {
            "from_client": bool(getattr(msg, "from_client", False)),
            **_content_payload(getattr(msg, "content", b"")),
        },
    }


def _udp_message_event(flow: Any, msg: Any) -> dict[str, Any]:
    return {
        "kind": "udp_message",
        "protocol": "udp",
        "hook": "message",
        "flow_id": str(getattr(flow, "id", "")),
        "client": str(getattr(flow.client_conn, "address", "")),
        "server": str(getattr(flow.server_conn, "address", "")),
        "message": {
            "from_client": bool(getattr(msg, "from_client", False)),
            **_content_payload(getattr(msg, "content", b"")),
        },
    }


def _dns_request_event(flow: Any, names: list[str]) -> dict[str, Any]:
    return {
        "kind": "dns_request",
        "protocol": "dns",
        "hook": "request",
        "flow_id": str(getattr(flow, "id", "")),
        "questions": names,
    }


def _content_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        raw = content.encode()
    elif isinstance(content, bytes):
        raw = content
    elif isinstance(content, bytearray | memoryview):
        raw = bytes(content)
    elif content is None:
        raw = b""
    else:
        raw = str(content).encode()

    payload: dict[str, Any] = {
        "bytes": len(raw),
        "body_base64": base64.b64encode(raw).decode(),
    }
    with contextlib.suppress(UnicodeDecodeError):
        payload["text"] = raw.decode()
    return payload


def _hook_url() -> str:
    return os.getenv("ABRIDGE_HOOK_URL", "")


def _send_hook_event_if_enabled(event: dict[str, Any]) -> dict[str, Any]:
    hook_url = _hook_url()
    if not hook_url:
        return _continue_action()
    try:
        return _send_hook_event(event, hook_url=hook_url)
    except Exception as exc:
        _trace("proxy_event.error", kind=event.get("kind"), error=repr(exc))
        return _continue_action()


def _send_hook_event(event: dict[str, Any], *, hook_url: str) -> dict[str, Any]:
    timeout = float(os.getenv("ABRIDGE_FORWARD_TIMEOUT", "600"))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(hook_url, json=event)
    resp.raise_for_status()
    action = resp.json()
    if not isinstance(action, dict):
        raise ValueError("sandbox forwarder returned non-object JSON")
    return action


def _apply_http_action(flow: Any, action: dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    name = action.get("action")
    if name == "respond":
        response = action.get("response")
        if not isinstance(response, dict):
            _set_response(flow, 502, b"bad respond action from abridge host")
            return True
        _apply_response_envelope(flow, response)
        return True
    if name == "kill":
        flow.kill()
        return True
    if name == "error":
        response = action.get("response")
        if isinstance(response, dict):
            _apply_response_envelope(flow, response)
        else:
            _set_response(flow, int(action.get("status_code") or 502), b"abridge hook error")
        return True
    return False


def _continue_action() -> dict[str, str]:
    return {"action": "continue"}


def _respond_action(response: dict[str, Any]) -> dict[str, Any]:
    return {"action": "respond", "response": response}


def _error_action(body: dict[str, Any], *, status_code: int = 502) -> dict[str, Any]:
    return _respond_action(_json_envelope(body, status_code=status_code)) | {"action": "error"}


def _apply_response_envelope(flow: Any, envelope: dict[str, Any]) -> None:
    raw_body = envelope.get("body_base64")
    if isinstance(raw_body, str):
        body = base64.b64decode(raw_body)
    else:
        body = str(envelope.get("body", "")).encode()

    headers = envelope.get("headers") or {}
    if not isinstance(headers, dict):
        headers = {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "application/octet-stream")
    status_code = int(envelope.get("status_code") or 200)
    _set_response(flow, status_code, body, content_type=content_type)
    for key, value in headers.items():
        if str(key).lower() == "content-type":
            continue
        flow.response.headers[str(key)] = str(value)


def _response_envelope(
    status_code: int,
    body: bytes,
    *,
    content_type: str = "text/plain",
) -> dict[str, Any]:
    return {
        "status_code": status_code,
        "headers": {
            "content-type": content_type,
            "content-length": str(len(body)),
        },
        "body_base64": base64.b64encode(body).decode(),
    }


def _json_envelope(body: dict[str, Any], *, status_code: int = 200) -> dict[str, Any]:
    payload = json.dumps(body, separators=(",", ":")).encode()
    return _response_envelope(status_code, payload, content_type="application/json")


def _count_tokens(body: dict[str, Any]) -> dict[str, int]:
    messages = body.get("messages") or []
    return {"input_tokens": len(json.dumps(messages, separators=(",", ":"))) // 4}


def _call_openai_compatible(body: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "authorization": f"Bearer {_openai_api_key()}",
        "content-type": "application/json",
        "accept": "application/json",
    }
    timeout = float(os.getenv("ABRIDGE_OPENAI_TIMEOUT", "120"))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(_openai_chat_completions_url(), headers=headers, json=body)
    resp.raise_for_status()
    value = resp.json()
    if not isinstance(value, dict):
        raise ValueError("openai-compatible upstream returned non-object JSON")
    return value


def _openai_chat_completions_url(base_url: str | None = None) -> str:
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid OPENAI_BASE_URL: {base_url}")
    return f"{parsed.scheme}://{parsed.netloc}{_join_url_path(parsed.path, '/chat/completions')}"


def _openai_api_key() -> str:
    return os.getenv("ABRIDGE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def _json_request_body(flow: Any) -> dict[str, Any]:
    raw = _message_text(flow.request)
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object request body")
    return value


def _message_text(message: Any) -> str:
    try:
        return message.get_text(strict=False)
    except TypeError:
        return message.get_text()


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _set_response(flow: Any, status_code: int, body: bytes, *, content_type: str = "text/plain") -> None:
    http = _mitm_http()
    flow.response = http.Response.make(status_code, body, {"content-type": content_type})


def _mitm_http() -> Any:
    global _MITM_HTTP
    if _MITM_HTTP is None:
        _MITM_HTTP = importlib.import_module("mitmproxy.http")
    return _MITM_HTTP


def _csv_env(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]


def _host_matches(host: str, patterns: list[str]) -> bool:
    host_without_port = host.rsplit(":", 1)[0] if ":" in host and not host.startswith("[") else host
    return any(
        host == pattern
        or host.endswith(f".{pattern}")
        or host_without_port == pattern
        or host_without_port.endswith(f".{pattern}")
        for pattern in patterns
    )


def _join_url_path(base_path: str, suffix: str) -> str:
    left = base_path.rstrip("/")
    right = suffix.lstrip("/")
    if not left:
        return f"/{right}"
    return f"{left}/{right}"


async def _wait_uvicorn_started(server: uvicorn.Server, configured_port: int) -> int:
    for _ in range(200):
        if server.started and server.servers:
            break
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("abridge mitm forwarder did not start within 10s")

    bound_port = configured_port
    for srv in server.servers:
        for sock in srv.sockets:
            bound_port = int(sock.getsockname()[1])
            break
        if bound_port:
            break
    return bound_port


async def _wait_port(host: str, port: int, process: asyncio.subprocess.Process) -> None:
    for _ in range(200):
        if process.returncode is not None:
            raise RuntimeError(f"mitmproxy exited early with code {process.returncode}")
        with contextlib.suppress(OSError):
            with socket.create_connection((host, port), timeout=0.2):
                return
        await asyncio.sleep(0.05)
    raise RuntimeError(f"mitmproxy did not listen on {host}:{port} within 10s")


def _free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


async def _drain_mitm_logs(stream: asyncio.StreamReader) -> None:
    async for raw in stream:
        text = raw.decode(errors="replace").rstrip()
        if text:
            logger.info("mitmproxy: %s", text)


def _mitmdump_args(argv: Sequence[str] | None = None) -> list[str]:
    args = list(argv if argv is not None else sys.argv[1:])
    if any(arg in {"-h", "--help", "--version", "--options", "--commands"} for arg in args):
        return args
    if not _has_option(args, "--mode", "-m"):
        args = ["--mode", os.getenv("ABRIDGE_MITM_MODE", "wireguard"), *args]
    if not _has_option(args, "--scripts", "-s"):
        args = ["-s", str(Path(__file__).resolve()), *args]
    return args


def _has_option(args: Sequence[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args for name in names)


def _trace(event: str, **fields: Any) -> None:
    if os.getenv("ABRIDGE_TRACE", "1") in {"0", "false", "False", "no"}:
        return
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


__all__ = [
    "HookForwarder",
    "OpenAIForwarder",
    "ProxyHandle",
    "dns_request",
    "main",
    "request",
    "response",
    "start_proxy",
    "stop_proxy",
    "tcp_message",
    "udp_message",
    "websocket_message",
]
