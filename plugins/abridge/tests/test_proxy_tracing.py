"""Verify the proxy emits a `trace.span("llm.request", ...)` per call.

The proxy wraps every captured LLM request in a span, attaches
phase events (`capture.request`, `translate.start/end`,
`upstream.start/end`), and decorates the span with usage attributes
on completion. This test installs a recording `Processor`, drives
one Anthropic and one OpenAI request through the proxy (against a
fake upstream HTTP server), and asserts the captured span tree
matches the expected shape.
"""

from __future__ import annotations

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
    start_proxy,
    stop_proxy,
)

from agentix import trace

# ── fake upstream ────────────────────────────────────────────────────────


class _Upstream(BaseHTTPRequestHandler):
    last_body: dict[str, Any] = {}

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        _Upstream.last_body = json.loads(self.rfile.read(length))
        blob = json.dumps(
            {
                "id": "chatcmpl-trace",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi"},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            }
        ).encode()
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


# ── recording trace processor ─────────────────────────────────────────────


class _RecordingProcessor(trace.Processor):
    def __init__(self) -> None:
        self.starts: list[trace.Span] = []
        self.ends: list[trace.Span] = []

    def on_trace_start(self, t: trace.Trace) -> None:
        return None

    def on_trace_end(self, t: trace.Trace) -> None:
        return None

    def on_span_start(self, s: trace.Span) -> None:
        self.starts.append(s)

    def on_span_end(self, s: trace.Span) -> None:
        self.ends.append(s)

    def force_flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


# ── in-process SIO pipe (mirrors test_proxy_roundtrip) ────────────────────


class _DirectSIO:
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
    import agentix as agentix_mod

    monkeypatch.setattr(agentix_mod, "register_namespace", lambda ns: None)
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
    pipe = _DirectSIO(namespace=sandbox_ns, host=host)

    async def sandbox_emit(event: str, data: Any = None) -> None:
        await pipe.emit(event, data)

    async def host_emit(event: str, data: Any = None, **_: Any) -> Any:
        await pipe.emit(event, data)

    monkeypatch.setattr(sandbox_ns, "emit", sandbox_emit)
    monkeypatch.setattr(host, "emit", host_emit)

    recorder = _RecordingProcessor()
    trace.add_processor(recorder)
    try:
        yield {
            "handle": handle,
            "store": store,
            "recorder": recorder,
        }
    finally:
        trace.remove_processor(recorder)
        await stop_proxy(handle)


# ── tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anthropic_request_emits_llm_request_span(wired) -> None:
    handle = wired["handle"]
    recorder: _RecordingProcessor = wired["recorder"]

    async with httpx.AsyncClient(base_url=handle.anthropic_base_url, timeout=10) as c:
        r = await c.post(
            "/v1/messages",
            json={
                "model": "claude-3-haiku",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
    assert r.status_code == 200

    # Exactly one llm.request span, ended.
    [ended] = [s for s in recorder.ends if s.name == "llm.request"]
    assert ended.status == "ok"
    assert ended.attrs["family"] == "anthropic.messages"
    assert ended.attrs["model"] == "claude-3-haiku"
    assert ended.attrs["request_path"] == "/v1/messages"
    assert ended.attrs["stream"] is False
    assert ended.attrs["usage.prompt_tokens"] == 9
    assert ended.attrs["usage.completion_tokens"] == 3
    assert ended.attrs["usage.total_tokens"] == 12
    assert ended.attrs["duration_ms"] >= 0

    event_names = [e.name for e in ended.events]
    # Phase events fire in order.
    assert event_names == [
        "capture.request",
        "translate.start",
        "translate.end",
        "upstream.start",
        "upstream.end",
        "translate.start",
        "translate.end",
    ]
    # Translation events carry direction attrs.
    directions = [
        e.attributes.get("direction")
        for e in ended.events
        if e.name in ("translate.start", "translate.end")
    ]
    assert directions == [
        "anthropic->openai",
        "anthropic->openai",
        "openai->anthropic",
        "openai->anthropic",
    ]


@pytest.mark.asyncio
async def test_openai_request_skips_translate_events(wired) -> None:
    handle = wired["handle"]
    recorder: _RecordingProcessor = wired["recorder"]

    async with httpx.AsyncClient(base_url=handle.openai_base_url, timeout=10) as c:
        r = await c.post(
            "/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ping"}]},
        )
    assert r.status_code == 200

    [ended] = [s for s in recorder.ends if s.name == "llm.request"]
    assert ended.attrs["family"] == "openai.chat.completions"
    event_names = [e.name for e in ended.events]
    # No anthropic translation legs on the OpenAI path.
    assert event_names == [
        "capture.request",
        "upstream.start",
        "upstream.end",
    ]


@pytest.mark.asyncio
async def test_upstream_error_marks_span_failed(wired, monkeypatch) -> None:
    handle = wired["handle"]
    recorder: _RecordingProcessor = wired["recorder"]

    # Patch the host's upstream POST to raise — covers the
    # `result["error"]` branch in `_handle_request`.
    sandbox_ns = proxy_mod._get_namespace()
    real_emit = sandbox_ns.emit

    async def failing_pipe_emit(event: str, data: Any = None) -> None:
        if event == "llm_call":
            sio_request_id = data.get("request_id") if isinstance(data, dict) else None
            await sandbox_ns._on_reply_success(
                {
                    "request_id": sio_request_id,
                    "value": {"error": {"message": "boom", "status_code": 500}},
                }
            )
            return
        await real_emit(event, data)

    monkeypatch.setattr(sandbox_ns, "emit", failing_pipe_emit)

    async with httpx.AsyncClient(base_url=handle.openai_base_url, timeout=10) as c:
        r = await c.post(
            "/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]},
        )
    assert r.status_code == 500

    [ended] = [s for s in recorder.ends if s.name == "llm.request"]
    assert ended.status == "error"
    assert ended.error is not None
    assert ended.error.message == "boom"
    assert ended.error.data == {"kind": "upstream", "status_code": 500}
    error_events = [e for e in ended.events if e.name == "upstream.error"]
    assert error_events and error_events[0].attributes["error"] == "boom"
