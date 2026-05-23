"""Sandbox proxy + host client integration test.

Spins up:

  * a fake "upstream" OpenAI-compatible server (just an HTTP server
    returning a canned Chat Completions response),
  * the host-side `OpenAICompatibleClient` pointing at that fake
    upstream,
  * the sandbox-side proxy from `agentix.bridge.proxy`,
  * a thin SIO wiring that ferries the proxy's events to the
    `OpenAICompatibleClient` and back.

Then issues two requests through the proxy — one in Anthropic format,
one in OpenAI format — and asserts they round-trip and produce
`CompletionRecord`s with correct usage.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import agentix.bridge.proxy as proxy_mod
import httpx
import pytest
import pytest_asyncio
from agentix.bridge import (
    InMemoryStore,
    OpenAICompatibleClient,
    detect,
    start_proxy,
    stop_proxy,
)
from agentix.bridge.detection import ApiFamily

# ── fake upstream OpenAI-compatible server ────────────────────────────────


class _Upstream(BaseHTTPRequestHandler):
    last_body: dict[str, Any] = {}

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        _Upstream.last_body = json.loads(raw)
        body = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello from upstream"},
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
        }
        blob = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *_: Any) -> None:
        return


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_upstream() -> Iterator[str]:
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _Upstream)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=2)


# ── inline SIO wiring (no real Socket.IO server needed) ───────────────────


class _DirectSIO:
    """Single-process pipe between the proxy's emit() and the host's on_*.

    The proxy uses `agentix.bridge.proxy._SandboxNamespace`, which is a
    subclass of `agentix.Namespace`. `agentix.Namespace` registers with
    the runtime SIO server in production; here we monkeypatch its
    `emit()` to fan out directly to the host client.

    For round-trip events the proxy calls `request()` which internally
    awaits a reply on `<event>:result`; we satisfy that by routing the
    host's reply back to the namespace's `_on_reply_success`.
    """

    def __init__(self, namespace: Any, host: OpenAICompatibleClient) -> None:
        self._ns = namespace
        self._host = host

    async def emit(self, event: str, data: Any = None) -> None:
        if event.endswith(":result"):
            await self._ns._on_reply_success(data)
            return
        if event.endswith(":error"):
            await self._ns._on_reply_error(data)
            return
        handler = getattr(self._host, f"on_{event}", None)
        if handler is None:
            return
        await handler(data)


@pytest_asyncio.fixture
async def wired(fake_upstream: str, monkeypatch):
    """Wire a sandbox proxy to a host client in-process.

    The proxy's `_get_namespace()` normally calls
    `agentix.register_namespace`, which requires the sandbox runtime
    worker pipe. For unit tests we monkeypatch that to a no-op and
    install our own emit shim so events fan out directly to the host
    coroutine.
    """
    import agentix as agentix_mod

    monkeypatch.setattr(agentix_mod, "register_namespace", lambda ns: None)
    # Reset cached namespace singleton across tests.
    monkeypatch.setattr(proxy_mod, "_namespace_singleton", None)

    handle = await start_proxy()

    store = InMemoryStore()
    host = OpenAICompatibleClient(
        base_url=fake_upstream,
        api_key="test-key",
        model="upstream-model",
        store=store,
    )

    sandbox_ns = proxy_mod._get_namespace()
    sio = _DirectSIO(namespace=sandbox_ns, host=host)

    async def sandbox_emit(event: str, data: Any = None) -> None:
        await sio.emit(event, data)

    async def host_emit(event: str, data: Any = None, **_: Any) -> Any:
        await sio.emit(event, data)

    monkeypatch.setattr(sandbox_ns, "emit", sandbox_emit)
    monkeypatch.setattr(host, "emit", host_emit)

    try:
        yield {"handle": handle, "host": host, "store": store, "upstream": fake_upstream}
    finally:
        await stop_proxy(handle)


# ── tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_request_routes_through_proxy(wired) -> None:
    handle = wired["handle"]
    store: InMemoryStore = wired["store"]

    async with httpx.AsyncClient(base_url=handle.anthropic_base_url, timeout=10) as c:
        r = await c.post(
            "/v1/messages",
            json={
                "model": "claude-3-haiku",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "claude-3-haiku"  # response carries the agent's model name
    assert body["content"][0]["text"] == "hello from upstream"
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 4}

    # Upstream saw the *translated* body with our overridden model.
    assert _Upstream.last_body["model"] == "upstream-model"
    assert _Upstream.last_body["messages"][-1]["content"] == "hi"

    # A CompletionRecord landed in the host store.
    await asyncio.sleep(0.05)  # let the fire-and-forget emit drain
    records = store.snapshot()
    assert len(records) == 1
    rec = records[0]
    assert rec.family is ApiFamily.ANTHROPIC_MESSAGES
    assert rec.usage.prompt_tokens == 7
    assert rec.usage.completion_tokens == 4


@pytest.mark.asyncio
async def test_openai_request_routes_through_proxy(wired) -> None:
    handle = wired["handle"]
    store: InMemoryStore = wired["store"]

    async with httpx.AsyncClient(base_url=handle.openai_base_url, timeout=10) as c:
        r = await c.post(
            "/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello from upstream"

    await asyncio.sleep(0.05)
    records = store.snapshot()
    assert len(records) == 1
    assert records[0].family is ApiFamily.OPENAI_CHAT_COMPLETIONS


def test_export_environ_shape() -> None:
    handle = proxy_mod.ProxyHandle(
        proxy_id="x",
        url="http://127.0.0.1:8000",
        port=8000,
        anthropic_base_url="http://127.0.0.1:8000",
        openai_base_url="http://127.0.0.1:8000/v1",
    )
    env = proxy_mod.export_environ(handle)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:8000/v1"
    assert env["ANTHROPIC_API_KEY"]
    assert env["OPENAI_API_KEY"]


def test_detect_anthropic_count_tokens_path() -> None:
    assert detect("/v1/messages/count_tokens") is ApiFamily.ANTHROPIC_COUNT_TOKENS
