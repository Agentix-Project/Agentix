"""Compatibility tests for the external ``cc_convert_sidecar`` preset.

The required test suite uses a small executable probe to verify the preset's
process and environment wiring without downloading an unversioned external
binary.  The Anthropic↔OpenAI compatibility test uses the real Rust binary
when a caller supplies one through ``CC_CONVERT_SIDECAR_BIN`` (or ``PATH``).
It remains optional until that binary has a published source and pinned
revision that CI can build reproducibly.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import httpx
import pytest
from agentix.bridge import Forward, Request, Sidecar
from agentix.bridge.sidecars import cc_convert_sidecar

BIN = os.environ.get("CC_CONVERT_SIDECAR_BIN") or shutil.which("cc_convert_sidecar")

REQUIRES_REAL_CC_CONVERT = pytest.mark.skipif(
    not BIN,
    reason="external cc_convert_sidecar binary not available (set CC_CONVERT_SIDECAR_BIN)",
)

PRESET_PROBE = r'''
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

CONFIG = {
    "listen_addr": os.environ["CC_CONVERT_LISTEN_ADDR"],
    "upstream_url": os.environ["CC_CONVERT_UPSTREAM_URL"],
    "upstream_key": os.environ.get("CC_CONVERT_UPSTREAM_API_KEY"),
    "litellm_compat": os.environ.get("CC_CONVERT_LITELLM_COMPAT"),
}

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps(CONFIG).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass

host, port = CONFIG["listen_addr"].rsplit(":", 1)
HTTPServer((host, int(port)), H).serve_forever()
'''

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


async def test_cc_convert_sidecar_preset_wires_process_environment(tmp_path) -> None:
    """The preset must pass the dynamically selected port and credentials."""
    probe = tmp_path / "cc_convert_sidecar_probe"
    probe.write_text(f"#!{sys.executable}\n{PRESET_PROBE}")
    probe.chmod(0o755)

    async with cc_convert_sidecar(
        binary=str(probe),
        upstream_url="https://openai.example/v1/chat/completions",
        upstream_key="test-key",
        litellm_compat=True,
        ready_timeout=5.0,
    ) as side_url:
        async with httpx.AsyncClient() as client:
            response = await client.get(side_url + "/config")
            response.raise_for_status()
            config = response.json()

    assert config == {
        "listen_addr": side_url.removeprefix("http://"),
        "upstream_url": "https://openai.example/v1/chat/completions",
        "upstream_key": "test-key",
        "litellm_compat": "1",
    }


@REQUIRES_REAL_CC_CONVERT
async def test_anthropic_through_cc_convert_sidecar(tmp_path) -> None:
    assert BIN is not None
    mock = tmp_path / "mock_openai.py"
    mock.write_text(MOCK_OPENAI)

    async with Sidecar(command=[sys.executable, str(mock), "{port}"]) as mock_url:
        async with cc_convert_sidecar(
            binary=BIN,
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

                # ── buffered SSE compatibility ──
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
