"""End-to-end: Anthropic agent → abridge Forward → cc_convert sidecar →
(mock) OpenAI upstream → translated Anthropic back.

This proves the redesigned Layer-1 path with ZERO translation code in
abridge: the real Rust `cc_convert_sidecar` does the Anthropic↔OpenAI
work; abridge only ferries bytes. A tiny mock OpenAI server stands in for
the upstream so the test needs no network and no real LLM.

Skipped unless the sidecar binary is available — set
`CC_CONVERT_SIDECAR_BIN` to its path (or put `cc_convert_sidecar` on
PATH). Build it with `cargo build -p cc_convert_sidecar`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest
from agentix.bridge import Forward, Request, Sidecar
from agentix.bridge.sidecars import cc_convert_sidecar

BIN = os.environ.get("CC_CONVERT_SIDECAR_BIN") or shutil.which("cc_convert_sidecar")

pytestmark = pytest.mark.skipif(
    not BIN,
    reason="cc_convert_sidecar binary not available (set CC_CONVERT_SIDECAR_BIN)",
)

# Mock OpenAI Chat Completions upstream: 200 on GET (health), an OpenAI
# completion on POST — streamed SSE when the request asks for it.
MOCK_OPENAI = r'''
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer

NONSTREAM = {
    "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": "mock",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hello from upstream"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}
CHUNKS = [
    {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}]},
    {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
     "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]},
    {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for c in CHUNKS:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n"); self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            payload = json.dumps(NONSTREAM).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


async def test_anthropic_through_cc_convert_sidecar(tmp_path) -> None:
    mock = tmp_path / "mock_openai.py"
    mock.write_text(MOCK_OPENAI)

    async with Sidecar(command=[sys.executable, str(mock), "{port}"]) as mock_url:
        async with cc_convert_sidecar(
            binary=str(BIN),
            upstream_url=mock_url + "/v1/chat/completions",
        ) as side_url:
            fwd = Forward(side_url, paths=["/v1/messages"])
            handler = fwd.abridge_routes()["/v1/messages"]
            try:
                # ── non-streaming: OpenAI completion → Anthropic message ──
                resp = await handler(
                    Request(
                        "/v1/messages",
                        {
                            "model": "claude-3-haiku",
                            "max_tokens": 50,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                )
                assert resp.media_type == "application/json"
                body = json.loads(resp.body)
                assert body["type"] == "message"
                assert body["role"] == "assistant"
                assert body["content"][0]["type"] == "text"
                assert body["content"][0]["text"] == "hello from upstream"
                assert body["stop_reason"] == "end_turn"
                assert body["usage"]["input_tokens"] == 5
                assert body["usage"]["output_tokens"] == 3

                # ── streaming: OpenAI SSE chunks → Anthropic SSE events ──
                sresp = await handler(
                    Request(
                        "/v1/messages",
                        {
                            "model": "claude-3-haiku",
                            "max_tokens": 50,
                            "stream": True,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    )
                )
                assert sresp.media_type == "text/event-stream"
                assert b"event: message_start" in sresp.body
                assert b"event: message_stop" in sresp.body
                assert b"hello" in sresp.body
            finally:
                await fwd.aclose()
