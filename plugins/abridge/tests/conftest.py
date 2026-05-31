"""Shared in-process test harness for abridge.

Wires a sandbox-side proxy to a host-side `Bridge`
without a real Socket.IO server: a fake OpenAI-compatible upstream, and
a `_DirectSIO` shim that fans the proxy's `emit()`/`request()` straight
into the host coroutine and back. The `wired` fixture is the entry point
used across the test modules.
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
from agentix.bridge import Bridge, InMemoryStore, start_proxy, stop_proxy

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
    """Single-process pipe between the proxy's emit() and the host's on_*."""

    def __init__(self, namespace: Any, host: Bridge) -> None:
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
    """Wire a sandbox proxy (session `sess-test`) to a host client in-process.

    Yields `{handle, host, store, upstream}` — `upstream` is the
    `_Upstream` handler class, whose `last_body`/`last_headers` record
    what the upstream actually received.
    """
    import agentix as agentix_mod

    monkeypatch.setattr(agentix_mod, "register_namespace", lambda ns: None)
    monkeypatch.setattr(proxy_mod, "_namespace_singleton", None)

    handle = await start_proxy(session_id="sess-test")

    store = InMemoryStore()
    host = Bridge(
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
        yield {"handle": handle, "host": host, "store": store, "upstream": _Upstream}
    finally:
        await stop_proxy(handle)
