"""End-to-end: serve mode composed onto a token-recording session gateway.

`agentix-bridge-serve --tito-url ...` builds, per caller session,
`AnthropicToOpenAI(SessionForward(tito_url).handler(), model=...)` — the
agent speaks Anthropic, the gateway receives the OpenAI chat body on its
session-scoped route and owns tokens + recording. A tiny in-process HTTP
server plays the gateway (create-session + session-scoped chat completions,
the same wire shapes as the real TITO gateway), so the whole path is real:
FastAPI serve app -> translation -> SessionForward -> real httpx -> gateway.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

import pytest
from agentix.bridge.serve import _build_parser, _client_factory, build_session_app, session_id_for
from fastapi.testclient import TestClient


class _FakeTito(BaseHTTPRequestHandler):
    """Session-scoped fake gateway: POST /sessions mints ids; the chat route
    records the exact OpenAI body + identity headers it received."""

    sessions: list[str] = []
    chat_calls: list[dict[str, Any]] = []  # {"session", "body", "headers"}

    @classmethod
    def reset(cls) -> None:
        cls.sessions = []
        cls.chat_calls = []

    def do_POST(self) -> None:  # noqa: N802 - http.server convention
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path == "/sessions":
            session_id = f"tito-sess-{len(_FakeTito.sessions)}"
            _FakeTito.sessions.append(session_id)
            self._json(200, {"session_id": session_id})
            return
        parts = self.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "sessions" and self.path.endswith("/v1/chat/completions"):
            _FakeTito.chat_calls.append({
                "session": parts[1],
                "body": json.loads(raw),
                "headers": {k.lower(): v for k, v in self.headers.items()},
            })
            self._json(200, {
                "id": "chatcmpl-tito", "object": "chat.completion", "model": "qwen3-4b",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello from tito"},
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            })
            return
        self._json(404, {"error": f"no route {self.path}"})

    def _json(self, status: int, body: dict) -> None:
        blob = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *_: Any) -> None:
        return


@pytest.fixture
def fake_tito() -> Iterator[str]:
    _FakeTito.reset()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), _FakeTito)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


_ANTHROPIC_BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 64,
    "system": "be brief",
    "messages": [{"role": "user", "content": "hi"}],
}


def _tito_app(fake_tito: str, *extra_args: str):
    args = _build_parser().parse_args(["--tito-url", fake_tito, "--upstream-model", "qwen3-4b", *extra_args])
    return build_session_app(_client_factory(args))


def test_tito_composition_end_to_end(fake_tito: str) -> None:
    tc = TestClient(_tito_app(fake_tito))

    r = tc.post("/v1/messages", json=_ANTHROPIC_BODY, headers={"x-api-key": "rollout-1"})
    assert r.status_code == 200

    # Agent side: a faithful Anthropic response.
    body = r.json()
    assert body["role"] == "assistant"
    assert body["content"] == [{"type": "text", "text": "hello from tito"}]
    assert body["model"] == "claude-sonnet-4-5"  # the agent's model id echoes back
    assert body["usage"] == {"input_tokens": 11, "output_tokens": 3}

    # Gateway side: one session created, the chat body is OpenAI-shaped,
    # forced non-streaming (the token recorder needs the full completion),
    # model overridden to what the engine serves.
    (call,) = _FakeTito.chat_calls
    assert call["session"] == _FakeTito.sessions[0]
    assert call["body"]["stream"] is False
    assert call["body"]["model"] == "qwen3-4b"
    assert call["body"]["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["body"]["messages"][1] == {"role": "user", "content": "hi"}

    # Identity stamping: the gateway groups by x-session-id (its own session
    # id — SessionForward adopts the gateway-assigned id) + per-call
    # x-request-id.
    assert call["headers"]["x-session-id"] == _FakeTito.sessions[0]
    assert call["headers"]["x-request-id"]


def test_tito_composition_one_gateway_session_per_caller_key(fake_tito: str) -> None:
    tc = TestClient(_tito_app(fake_tito))
    assert tc.post("/v1/messages", json=_ANTHROPIC_BODY, headers={"x-api-key": "rollout-a"}).status_code == 200
    assert tc.post("/v1/messages", json=_ANTHROPIC_BODY, headers={"x-api-key": "rollout-a"}).status_code == 200
    assert tc.post("/v1/messages", json=_ANTHROPIC_BODY, headers={"x-api-key": "rollout-b"}).status_code == 200

    assert len(_FakeTito.sessions) == 2  # one gateway session per caller key
    assert [c["session"] for c in _FakeTito.chat_calls] == [
        _FakeTito.sessions[0], _FakeTito.sessions[0], _FakeTito.sessions[1],
    ]


def test_tito_composition_with_record_dir_joins_rows_to_gateway_calls(fake_tito: str, tmp_path) -> None:
    """--record-dir in tito mode: the message-level Recorder row and the
    gateway's x-request-id share one id, and the row's session_id is the
    caller-derived serve session (the row->record join key set)."""
    tc = TestClient(_tito_app(fake_tito, "--record-dir", str(tmp_path)))
    r = tc.post("/v1/messages", json=_ANTHROPIC_BODY, headers={"x-api-key": "rollout-1"})
    assert r.status_code == 200

    serve_session = session_id_for("rollout-1")
    (row,) = [json.loads(line) for line in (tmp_path / f"{serve_session}.jsonl").read_text().splitlines()]
    (call,) = _FakeTito.chat_calls
    assert row["session_id"] == serve_session
    assert row["request_id"] == call["headers"]["x-request-id"]
    assert row["path"] == "/v1/messages"
    assert row["request"] == _ANTHROPIC_BODY  # the agent-side (Anthropic) shape
    assert row["response"]["body"]["content"] == [{"type": "text", "text": "hello from tito"}]
    # No file for the route-enumeration probe session.
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{serve_session}.jsonl"]


def test_tito_streaming_agent_gets_replayed_sse(fake_tito: str) -> None:
    """stream:true agents get the locally rendered SSE replay while the
    gateway still saw a non-streaming call."""
    tc = TestClient(_tito_app(fake_tito))
    r = tc.post(
        "/v1/messages",
        json={**_ANTHROPIC_BODY, "stream": True},
        headers={"x-api-key": "rollout-1"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "hello from tito" in r.text
    assert _FakeTito.chat_calls[0]["body"]["stream"] is False


def test_serve_mode_flags_are_mutually_exclusive_and_required(monkeypatch) -> None:
    from agentix.bridge.serve import main

    for var in ("OPENAI_BASE_URL", "TITO_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit):
        main(["--upstream-base-url", "http://engine:8000/v1", "--tito-url", "http://tito:30000"])
    with pytest.raises(SystemExit):
        main([])  # neither mode selected
