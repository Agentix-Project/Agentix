"""Tests for the Layer-1 primitives: `Forward` + `Sidecar`.

`Forward` is the schema-agnostic JSON POST handler that sends an agent's
decoded JSON object to a sidecar URL; `Sidecar` owns a local gateway
process's lifecycle. The unit tests mock httpx; the integration tests spawn
a tiny echo HTTP server as a real sidecar to prove the whole launch →
health → forward path locally, with no LLM in the loop.
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
    SessionForward,
    Sidecar,
    SidecarError,
    TunnelHandle,
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


async def test_forward_returns_buffered_sse_compatibility(monkeypatch) -> None:
    """SSE bytes and media type survive, but only after the body is buffered."""
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


async def test_forward_upstream_http_error_is_a_response(monkeypatch) -> None:
    fwd = Forward("http://side.car", paths=["/v1/messages"])

    async def fake_post(url, *, json, headers):
        return httpx.Response(503, content=b'{"error":"down"}', headers={"content-type": "application/json"})

    monkeypatch.setattr(fwd._client, "post", fake_post)
    resp = await fwd.abridge_routes()["/v1/messages"](_req("/v1/messages", {}))
    assert resp.status_code == 503
    assert resp.body == b'{"error":"down"}'


async def test_forward_network_error_is_503(monkeypatch) -> None:
    """Failure to reach the sidecar is 503 — distinct from a relayed upstream 502."""
    fwd = Forward("http://side.car", paths=["/v1/messages"])

    async def fake_post(url, *, json, headers):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(fwd._client, "post", fake_post)
    with pytest.raises(AbridgeError) as ei:
        await fwd.abridge_routes()["/v1/messages"](_req("/v1/messages", {}))
    assert ei.value.status_code == 503


async def test_forward_http_status_survives_tunnel_and_sio(monkeypatch) -> None:
    """A sidecar 503 stays a normal result across the complete wire path."""
    import agentix.bridge.proxy as proxy_mod

    import agentix as agentix_mod

    monkeypatch.setattr(agentix_mod, "register_namespace", lambda ns: None)
    monkeypatch.setattr(proxy_mod, "_namespace_singleton", None)

    fwd = Forward("http://side.car", paths=["/v1/messages"])

    async def fake_post(url, *, json, headers):
        return httpx.Response(
            503,
            content=b'{"error":"sidecar unavailable"}',
            headers={"content-type": "application/json"},
        )

    assert fwd._client is not None
    monkeypatch.setattr(fwd._client, "post", fake_post)
    host = Proxy(fwd)
    handle = await proxy_mod._start_tunnel(paths=list(host.paths))
    sandbox_ns = proxy_mod._get_namespace()

    async def sandbox_emit(event, data=None):
        await host.trigger_event(event, data)

    async def host_emit(event, data=None, **kwargs):
        if event.endswith(":result"):
            await sandbox_ns._on_reply_success(data)
        elif event.endswith(":error"):
            await sandbox_ns._on_reply_error(data)

    monkeypatch.setattr(sandbox_ns, "emit", sandbox_emit)
    monkeypatch.setattr(host, "emit", host_emit)

    try:
        async with httpx.AsyncClient(base_url=handle.url, timeout=10) as client:
            response = await client.post("/v1/messages", json={"model": "claude"})
        assert response.status_code == 503
        assert response.content == b'{"error":"sidecar unavailable"}'
    finally:
        await proxy_mod._stop_tunnel(handle=handle)
        await fwd.aclose()


class _CloseAwareClient:
    def __init__(self) -> None:
        self.close_calls = 0

    def abridge_routes(self):
        return {"/v1/messages": self.handle}

    async def handle(self, request: Request) -> ClientResponse:
        return ClientResponse.json({"ok": True})

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeSandbox:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.remote_calls = 0

    def register_namespace(self, namespace) -> None:
        pass

    async def remote(self, fn, **kwargs):
        self.remote_calls += 1
        if self.fail_start:
            raise RuntimeError("tunnel failed to start")
        return TunnelHandle(url="http://127.0.0.1:1", port=1)


async def test_proxy_session_closes_clients_once() -> None:
    client = _CloseAwareClient()
    proxy = Proxy(client)
    sandbox = _FakeSandbox()

    async with proxy.session(sandbox):
        assert client.close_calls == 0

    assert client.close_calls == 1
    await proxy.stop(sandbox)
    assert client.close_calls == 1


async def test_proxy_closes_clients_when_start_fails() -> None:
    client = _CloseAwareClient()
    proxy = Proxy(client)
    sandbox = _FakeSandbox(fail_start=True)

    with pytest.raises(RuntimeError, match="failed to start"):
        await proxy.start(sandbox)

    assert client.close_calls == 1
    await proxy.stop(sandbox)
    assert client.close_calls == 1


async def test_forward_pool_can_reopen_after_idempotent_close() -> None:
    fwd = Forward("http://side.car", paths=["/v1/messages"])
    original = fwd._client
    assert original is not None

    await fwd.aclose()
    await fwd.aclose()
    assert original.is_closed

    replacement = fwd._get_client()
    assert replacement is not original
    assert not replacement.is_closed
    await fwd.aclose()


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


# ── SessionForward (unit, mocked httpx) ───────────────────────────────


async def test_session_forward_creates_session_then_rewrites_path(monkeypatch) -> None:
    fwd = SessionForward("http://gw", paths=["/v1/chat/completions"])
    calls: list = []

    async def fake_post(url, *, json, headers):
        calls.append((url, json, headers))
        if url.endswith("/sessions"):
            return httpx.Response(200, content=b'{"session_id": "S9"}', headers={"content-type": "application/json"})
        return httpx.Response(200, content=b'{"ok": true}', headers={"content-type": "application/json"})

    assert fwd._client is not None
    monkeypatch.setattr(fwd._client, "post", fake_post)
    resp = await fwd.abridge_routes()["/v1/chat/completions"](
        _req("/v1/chat/completions", {"model": "qwen3-4b"})
    )

    assert isinstance(resp, ClientResponse)
    assert resp.status_code == 200 and resp.body == b'{"ok": true}'
    assert fwd.session_id == "S9"
    # First upstream call created the session; second routed into it by path.
    assert calls[0][0] == "http://gw/sessions"
    assert calls[1][0] == "http://gw/sessions/S9/v1/chat/completions"
    assert calls[1][1] == {"model": "qwen3-4b"}
    assert calls[1][2]["x-session-id"] == "S9"


async def test_session_forward_creates_session_once(monkeypatch) -> None:
    fwd = SessionForward("http://gw", paths=["/v1/chat/completions"])
    creates = 0

    async def fake_post(url, *, json, headers):
        nonlocal creates
        if url.endswith("/sessions"):
            creates += 1
            return httpx.Response(200, content=b'{"session_id": "S"}', headers={"content-type": "application/json"})
        return httpx.Response(200, content=b"{}", headers={"content-type": "application/json"})

    assert fwd._client is not None
    monkeypatch.setattr(fwd._client, "post", fake_post)
    handler = fwd.abridge_routes()["/v1/chat/completions"]
    await handler(_req("/v1/chat/completions", {}))
    await handler(_req("/v1/chat/completions", {}))
    assert creates == 1


async def test_session_forward_open_precreates_session(monkeypatch) -> None:
    fwd = SessionForward("http://gw", paths=["/v1/chat/completions"])

    async def fake_post(url, *, json, headers):
        return httpx.Response(200, content=b'{"session_id": "PRE"}', headers={"content-type": "application/json"})

    assert fwd._client is not None
    monkeypatch.setattr(fwd._client, "post", fake_post)
    assert await fwd.open() == "PRE"
    assert fwd.session_id == "PRE"


async def test_session_forward_create_failure_is_502(monkeypatch) -> None:
    fwd = SessionForward("http://gw", paths=["/v1/chat/completions"])

    async def fake_post(url, *, json, headers):
        return httpx.Response(500, content=b"boom")

    assert fwd._client is not None
    monkeypatch.setattr(fwd._client, "post", fake_post)
    with pytest.raises(AbridgeError) as ei:
        await fwd.open()
    assert ei.value.status_code == 502


async def test_session_forward_missing_id_field_is_502(monkeypatch) -> None:
    fwd = SessionForward("http://gw", paths=["/v1/chat/completions"])

    async def fake_post(url, *, json, headers):
        return httpx.Response(200, content=b'{"nope": 1}', headers={"content-type": "application/json"})

    assert fwd._client is not None
    monkeypatch.setattr(fwd._client, "post", fake_post)
    with pytest.raises(AbridgeError) as ei:
        await fwd.open()
    assert ei.value.status_code == 502


def test_session_forward_session_id_before_open_raises() -> None:
    fwd = SessionForward("http://gw", paths=["/v1/chat/completions"])
    with pytest.raises(RuntimeError):
        _ = fwd.session_id


async def test_session_forward_delete_session_reaps_and_resets(monkeypatch) -> None:
    fwd = SessionForward("http://gw", paths=["/v1/chat/completions"])
    deleted: list = []

    async def fake_post(url, *, json, headers):
        return httpx.Response(200, content=b'{"session_id": "S"}', headers={"content-type": "application/json"})

    async def fake_delete(url, *, headers):
        deleted.append(url)
        return httpx.Response(204)

    assert fwd._client is not None
    monkeypatch.setattr(fwd._client, "post", fake_post)
    monkeypatch.setattr(fwd._client, "delete", fake_delete)

    assert await fwd.open() == "S"
    await fwd.delete_session()
    assert deleted == ["http://gw/sessions/S"]
    # after reaping, the id is gone again — a premature read raises.
    with pytest.raises(RuntimeError):
        _ = fwd.session_id


# ── SessionForward (integration, real sidecar) ────────────────────────

SESSION_SERVER = """
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer

SID = "sess-LIVE"

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        if self.path == "/sessions":
            self.wfile.write(json.dumps({"session_id": SID}).encode())
        else:
            self.wfile.write(json.dumps({"path": self.path, "got": json.loads(body or b"{}")}).encode())

    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


async def test_session_forward_through_live_sidecar(tmp_path) -> None:
    script = tmp_path / "sess.py"
    script.write_text(SESSION_SERVER)
    async with Sidecar(command=[sys.executable, str(script), "{port}"]) as url:
        fwd = SessionForward(url, paths=["/v1/chat/completions"])
        try:
            resp = await fwd.abridge_routes()["/v1/chat/completions"](
                _req("/v1/chat/completions", {"model": "m"})
            )
            assert resp.status_code == 200
            assert fwd.session_id == "sess-LIVE"
            assert b'"path": "/sessions/sess-LIVE/v1/chat/completions"' in resp.body
            assert b'"model": "m"' in resp.body
        finally:
            await fwd.aclose()


# ── tunnel boundary + client env helpers ──────────────────────────────


async def test_tunnel_rejects_non_object_body(monkeypatch) -> None:
    """A present-but-non-object JSON body (an array) is a 400 at the tunnel,
    not a silent coercion to {}."""
    import agentix as agentix_mod
    import agentix.bridge.proxy as proxy_mod

    monkeypatch.setattr(agentix_mod, "register_namespace", lambda ns: None)
    monkeypatch.setattr(proxy_mod, "_namespace_singleton", None)

    handle = await proxy_mod._start_tunnel(paths=["/v1/messages"])
    try:
        async with httpx.AsyncClient(base_url=handle.url, timeout=10) as client:
            r = await client.post("/v1/messages", json=["not", "an", "object"])
        assert r.status_code == 400
    finally:
        await proxy_mod._stop_tunnel(handle=handle)


def test_openai_client_environ_bakes_in_v1() -> None:
    """OpenAIClient.environ mirrors the Anthropic ones and bakes in the /v1 suffix."""
    from agentix.bridge.clients import OpenAIClient

    c = OpenAIClient(base_url="https://up.stream/v1", api_key="real-key", model="gpt-4o")
    env = c.environ(TunnelHandle(url="http://127.0.0.1:9", port=9))
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9/v1"
    assert env["OPENAI_API_KEY"].startswith("sk-")


# ── composition: Convert (AnthropicToOpenAI) ∘ transport (Forward/SessionForward) ──

_OPENAI_COMPLETION = (
    b'{"id":"c1","object":"chat.completion","model":"qwen3-4b",'
    b'"choices":[{"index":0,"finish_reason":"stop",'
    b'"message":{"role":"assistant","content":"hi there"}}],'
    b'"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}'
)


def test_forward_handler_accessor() -> None:
    fwd = Forward("http://x", paths=["/v1/chat/completions"])
    assert callable(fwd.handler())  # sole path, no arg needed
    assert callable(fwd.handler("/v1/chat/completions"))
    with pytest.raises(ValueError):
        Forward("http://x", paths=["/a", "/b"]).handler()  # ambiguous → must name a path


async def test_anthropic_to_openai_translates_over_any_downstream() -> None:
    """The Convert layer is transport-blind: it translates Anthropic→OpenAI, calls
    the downstream Handler, and translates the OpenAI completion back to Anthropic."""
    from agentix.bridge.clients import AnthropicToOpenAI

    captured: dict = {}

    async def fake_downstream(request: Request) -> ClientResponse:
        captured["body"] = request.body
        return ClientResponse(body=_OPENAI_COMPLETION, media_type="application/json")

    conv = AnthropicToOpenAI(fake_downstream, model="qwen3-4b")
    resp = await conv.messages(
        _req("/v1/messages", {
            "model": "claude-3-5-sonnet",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "say hi"}],
        })
    )
    assert resp.status_code == 200
    assert b"hi there" in resp.body
    assert b'"role": "assistant"' in resp.body          # Anthropic response shape
    # the downstream saw an OpenAI-shaped, non-streaming body with the model override
    assert captured["body"]["model"] == "qwen3-4b"
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0]["role"] == "user"


async def test_anthropic_to_openai_composes_with_session_forward(monkeypatch) -> None:
    """End-to-end composition: Anthropic agent → AnthropicToOpenAI → SessionForward
    creates the session and rewrites the path → OpenAI completion → back to Anthropic.
    The converter never touches a session; the SessionForward never sees Anthropic."""
    from agentix.bridge.clients import AnthropicToOpenAI

    tito = SessionForward("http://gw", paths=["/v1/chat/completions"])
    calls: list = []

    async def fake_post(url, *, json, headers):
        calls.append(url)
        if url.endswith("/sessions"):
            return httpx.Response(200, content=b'{"session_id": "S1"}', headers={"content-type": "application/json"})
        return httpx.Response(200, content=_OPENAI_COMPLETION, headers={"content-type": "application/json"})

    assert tito._client is not None
    monkeypatch.setattr(tito._client, "post", fake_post)

    conv = AnthropicToOpenAI(tito.handler(), model="qwen3-4b")
    resp = await conv.messages(
        _req("/v1/messages", {
            "model": "claude",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "say hi"}],
        })
    )
    assert resp.status_code == 200
    assert b"hi there" in resp.body
    assert tito.session_id == "S1"
    assert calls[0] == "http://gw/sessions"
    assert calls[1] == "http://gw/sessions/S1/v1/chat/completions"
    await tito.aclose()


async def test_anthropic_to_openai_replays_remembered_assistant() -> None:
    """The lossy Anthropic round-trip drops reasoning_content + tool-call `index`, so
    a reconstructed assistant won't byte-match what a session backend stored. The
    converter remembers the exact downstream assistant and replays it verbatim on the
    next turn, keyed by the surviving tool-call ids."""
    from agentix.bridge.clients import AnthropicToOpenAI

    stored_assistant = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "\n\n",
        "tool_calls": [{
            "id": "call_X", "index": 0, "type": "function",
            "function": {"name": "python", "arguments": "{\"expression\": \"1+1\"}"},
        }],
    }
    sent: list = []

    async def fake_downstream(request: Request) -> ClientResponse:
        sent.append(request.body.get("messages"))
        return ClientResponse.json({
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": stored_assistant}],
            "model": "m", "usage": {},
        })

    conv = AnthropicToOpenAI(fake_downstream, model="m")
    await conv.messages(_req("/v1/messages", {
        "model": "c", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}],
    }))
    # turn 1: the agent resends the round-tripped assistant (no index, no reasoning_content)
    await conv.messages(_req("/v1/messages", {
        "model": "c", "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_X", "name": "python", "input": {"expression": "1+1"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_X", "content": "2"}]},
        ],
    }))

    asst = [m for m in sent[1] if m.get("role") == "assistant"]
    assert len(asst) == 1
    # value-equal to the stored assistant (incl. reasoning_content + index), not the
    # lossy reconstruction — so the backend's byte-match would pass.
    assert asst[0] == stored_assistant
    assert asst[0]["reasoning_content"] == "\n\n"
    assert asst[0]["tool_calls"][0]["index"] == 0
