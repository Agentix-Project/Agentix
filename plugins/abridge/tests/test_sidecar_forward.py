"""Tests for the Layer-1 primitives: `Forward` + `Sidecar`.

`Forward` is the protocol-blind handler that ferries an agent's request to
a sidecar URL; `Sidecar` owns a local gateway process's lifecycle. The
unit tests mock httpx; the integration tests spawn a tiny echo HTTP server
as a real sidecar to prove the whole launch → health → forward path
locally, with no LLM in the loop.
"""

from __future__ import annotations

import sys

import httpx
import pytest
from agentix.bridge import (
    AbridgeError,
    ClientResponse,
    Forward,
    Proxy,
    Request,
    Sidecar,
    SidecarError,
)

# A minimal HTTP server: 200 on any GET (health), echo JSON on POST.
ECHO_SERVER = """
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"echo": true}')

    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def _req(path: str, body: dict) -> Request:
    return Request(path=path, body=body)


# ── Forward (unit, mocked httpx) ──────────────────────────────────────


def test_forward_exposes_dynamic_routes() -> None:
    fwd = Forward("http://localhost:9999", paths=["/v1/messages", "/v1/messages/count_tokens"])
    assert set(fwd.abridge_routes()) == {"/v1/messages", "/v1/messages/count_tokens"}


def test_proxy_collects_forward_routes() -> None:
    proxy = Proxy(Forward("http://localhost:9999", paths=["/v1/messages"]))
    assert proxy.paths == ("/v1/messages",)


def test_forward_requires_paths() -> None:
    with pytest.raises(ValueError):
        Forward("http://localhost:9999", paths=[])


async def test_forward_posts_body_and_stamps_identity(monkeypatch) -> None:
    fwd = Forward("http://side.car", paths=["/v1/messages"], session_id="sess-1")
    captured: dict = {}

    async def fake_post(url, *, json, headers):
        captured.update(url=url, json=json, headers=headers)
        return httpx.Response(200, content=b'{"ok": true}', headers={"content-type": "application/json"})

    monkeypatch.setattr(fwd._client, "post", fake_post)
    resp = await fwd.abridge_routes()["/v1/messages"](_req("/v1/messages", {"model": "claude", "x": 1}))

    assert isinstance(resp, ClientResponse)
    assert resp.body == b'{"ok": true}'
    assert resp.media_type == "application/json"
    assert captured["url"] == "http://side.car/v1/messages"
    assert captured["json"] == {"model": "claude", "x": 1}
    assert captured["headers"]["x-session-id"] == "sess-1"
    assert "x-request-id" in captured["headers"]


async def test_forward_sse_passthrough(monkeypatch) -> None:
    fwd = Forward("http://side.car", paths=["/v1/messages"])

    async def fake_post(url, *, json, headers):
        return httpx.Response(
            200,
            content=b"event: message_start\ndata: {}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(fwd._client, "post", fake_post)
    resp = await fwd.abridge_routes()["/v1/messages"](_req("/v1/messages", {"stream": True}))
    assert resp.media_type == "text/event-stream"
    assert b"event: message_start" in resp.body


async def test_forward_upstream_error_preserves_status(monkeypatch) -> None:
    fwd = Forward("http://side.car", paths=["/v1/messages"])

    async def fake_post(url, *, json, headers):
        return httpx.Response(503, content=b'{"error":"down"}', headers={"content-type": "application/json"})

    monkeypatch.setattr(fwd._client, "post", fake_post)
    with pytest.raises(AbridgeError) as ei:
        await fwd.abridge_routes()["/v1/messages"](_req("/v1/messages", {}))
    assert ei.value.status_code == 503


async def test_forward_network_error_is_502(monkeypatch) -> None:
    fwd = Forward("http://side.car", paths=["/v1/messages"])

    async def fake_post(url, *, json, headers):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(fwd._client, "post", fake_post)
    with pytest.raises(AbridgeError) as ei:
        await fwd.abridge_routes()["/v1/messages"](_req("/v1/messages", {}))
    assert ei.value.status_code == 502


# ── Sidecar + Forward (integration, real subprocess) ──────────────────


async def test_sidecar_starts_and_tears_down(tmp_path) -> None:
    script = tmp_path / "srv.py"
    script.write_text(ECHO_SERVER)
    async with Sidecar(command=[sys.executable, str(script), "{port}"], health_path="/healthz") as url:
        async with httpx.AsyncClient() as c:
            assert (await c.get(url + "/healthz")).status_code == 200
    # After exit the process is gone — the port no longer accepts connections.
    with pytest.raises(httpx.HTTPError):
        async with httpx.AsyncClient(timeout=1.0) as c:
            await c.get(url + "/healthz")


async def test_sidecar_unhealthy_process_raises() -> None:
    with pytest.raises(SidecarError):
        async with Sidecar(
            command=[sys.executable, "-c", "import sys; sys.exit(1)"],
            ready_timeout=3.0,
        ):
            pass


async def test_forward_through_live_sidecar(tmp_path) -> None:
    script = tmp_path / "srv.py"
    script.write_text(ECHO_SERVER)
    async with Sidecar(command=[sys.executable, str(script), "{port}"]) as url:
        fwd = Forward(url, paths=["/v1/messages"])
        try:
            resp = await fwd.abridge_routes()["/v1/messages"](_req("/v1/messages", {"hi": 1}))
            assert resp.body == b'{"echo": true}'
            assert resp.media_type == "application/json"
        finally:
            await fwd.aclose()
