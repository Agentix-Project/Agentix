"""Shared in-process test harness for abridge.

Wires a sandbox-side tunnel to a host-side `Proxy` without a real
Socket.IO server. The `wired` fixture: a fake OpenAI-compatible
upstream + a `_DirectSIO` shim that fans the tunnel's `emit()` /
`request()` straight into the `Proxy.trigger_event` and back.

The default host setup is
`Proxy(AnthropicFromOpenAIClient(...), OpenAIClient(...))` — both
clients composed at the Proxy constructor — so the fixture exercises
`/v1/messages` (+ `/v1/messages/count_tokens`) and `/v1/chat/completions`
through the same handle.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import agentix.bridge.proxy as proxy_mod
import pytest
import pytest_asyncio
from agentix.bridge import Proxy
from agentix.bridge.clients import AnthropicFromOpenAIClient, OpenAIClient
from agentix.bridge.proxy import _start_tunnel, _stop_tunnel

# ── fake upstream OpenAI-compatible server ────────────────────────────────


class _Upstream(BaseHTTPRequestHandler):
    last_body: dict[str, Any] = {}
    last_headers: dict[str, str] = {}

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        _Upstream.last_body = json.loads(raw)
        _Upstream.last_headers = {k.lower(): v for k, v in self.headers.items()}
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "upstream-model",
            "choices": [
                {
                    "index": 0,
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
    """Single-process pipe between the tunnel's `emit()` and the host's
    `trigger_event`. Mirrors what `socketio` does over the wire."""

    def __init__(self, namespace: Any, host: Proxy) -> None:
        self._ns = namespace
        self._host = host

    async def emit(self, event: str, data: Any = None) -> None:
        if event.endswith(":result"):
            await self._ns._on_reply_success(data)
            return
        if event.endswith(":error"):
            await self._ns._on_reply_error(data)
            return
        await self._host.trigger_event(event, data)


@pytest_asyncio.fixture
async def wired(fake_upstream: str, monkeypatch):
    """Wire a sandbox tunnel (session `sess-test`) to a host `Proxy`
    in-process. Yields `{handle, host, upstream}`.

    The Proxy serves both Anthropic (`/v1/messages`,
    `/v1/messages/count_tokens`) and OpenAI (`/v1/chat/completions`).
    """
    import agentix as agentix_mod

    monkeypatch.setattr(agentix_mod, "register_namespace", lambda ns: None)
    monkeypatch.setattr(proxy_mod, "_namespace_singleton", None)

    afo = AnthropicFromOpenAIClient(
        base_url=fake_upstream, api_key="test-key", model="upstream-model",
        session_id="sess-test",
    )
    openai = OpenAIClient(
        base_url=fake_upstream, api_key="test-key", model="upstream-model",
        session_id="sess-test",
    )
    host = Proxy(afo, openai)

    handle = await _start_tunnel(paths=list(host.paths))

    sandbox_ns = proxy_mod._get_namespace()
    sio = _DirectSIO(namespace=sandbox_ns, host=host)

    async def sandbox_emit(event: str, data: Any = None) -> None:
        await sio.emit(event, data)

    async def host_emit(event: str, data: Any = None, **_: Any) -> Any:
        await sio.emit(event, data)

    monkeypatch.setattr(sandbox_ns, "emit", sandbox_emit)
    monkeypatch.setattr(host, "emit", host_emit)

    try:
        yield {"handle": handle, "host": host, "upstream": _Upstream}
    finally:
        await _stop_tunnel(handle=handle)
