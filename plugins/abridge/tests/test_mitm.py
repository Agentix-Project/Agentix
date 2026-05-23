from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from agentix.bridge import mitm


class _Message:
    def __init__(self, content: bytes = b"") -> None:
        self.headers: dict[str, str] = {}
        self.content = content

    def get_text(self, strict: bool = True) -> str:
        return self.content.decode()


class _Request(_Message):
    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__(json.dumps(body).encode())
        self.method = "POST"
        self.pretty_host = "api.anthropic.com"
        self.scheme = "https"
        self.host = "api.anthropic.com"
        self.port = 443
        self.path = "/v1/messages"


class _Response(_Message):
    def __init__(self, body: dict[str, Any], status_code: int = 200) -> None:
        super().__init__(json.dumps(body).encode())
        self.status_code = status_code


class _MadeResponse(_Message):
    def __init__(self, status_code: int, content: bytes, headers: dict[str, str]) -> None:
        super().__init__(content)
        self.status_code = status_code
        self.headers = dict(headers)


class _MitmHTTP:
    class Response:
        @staticmethod
        def make(status_code: int, content: bytes, headers: dict[str, str]) -> _MadeResponse:
            return _MadeResponse(status_code, content, headers)


class _Flow:
    def __init__(self, body: dict[str, Any]) -> None:
        self.request = _Request(body)
        self.response: _Response | None = None
        self.metadata: dict[str, Any] = {}
        self.killed = False

    def kill(self) -> None:
        self.killed = True


class _Conn:
    def __init__(self, address: tuple[str, int]) -> None:
        self.address = address


class _TcpMessage:
    def __init__(self, content: bytes, *, from_client: bool = True) -> None:
        self.content = content
        self.from_client = from_client


class _TcpFlow:
    def __init__(self, content: bytes) -> None:
        self.id = "tcp-1"
        self.client_conn = _Conn(("10.0.0.2", 5555))
        self.server_conn = _Conn(("10.0.0.3", 443))
        self.messages = [_TcpMessage(content)]


@pytest.fixture(autouse=True)
def fake_mitm_http(monkeypatch) -> None:
    monkeypatch.setattr(mitm, "_MITM_HTTP", _MitmHTTP)


def test_request_sends_http_proxy_event_to_sandbox_forwarder(monkeypatch) -> None:
    monkeypatch.setenv("ABRIDGE_TRACE", "0")
    monkeypatch.setenv("ABRIDGE_HOOK_URL", "http://127.0.0.1:4567/hook")
    flow = _Flow(
        {
            "model": "claude",
            "system": "be terse",
            "stream": True,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "name": "bash",
                    "description": "run command",
                    "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                }
            ],
        }
    )
    seen: dict[str, Any] = {}

    def fake_send(event: dict[str, Any], *, hook_url: str) -> dict[str, Any]:
        seen["hook_url"] = hook_url
        seen["event"] = event
        return {
            "action": "respond",
            "response": mitm._response_envelope(
                200,
                b"event: message_stop\n\n",
                content_type="text/event-stream",
            ),
        }

    monkeypatch.setattr(mitm, "_send_hook_event", fake_send)

    mitm.request(flow)

    assert seen["hook_url"] == "http://127.0.0.1:4567/hook"
    event = seen["event"]
    assert event["kind"] == "http_request"
    assert event["protocol"] == "http"
    assert event["hook"] == "request"
    assert event["request"]["path"] == "/v1/messages"
    assert event["request"]["json"]["stream"] is True
    assert flow.response.status_code == 200
    assert flow.response.headers["content-type"] == "text/event-stream"
    assert flow.response.content == b"event: message_stop\n\n"


def test_tcp_message_sends_protocol_neutral_proxy_event(monkeypatch) -> None:
    monkeypatch.setenv("ABRIDGE_TRACE", "0")
    monkeypatch.setenv("ABRIDGE_HOOK_URL", "http://127.0.0.1:4567/hook")
    flow = _TcpFlow(b"\x00abc")
    seen: dict[str, Any] = {}

    def fake_send(event: dict[str, Any], *, hook_url: str) -> dict[str, Any]:
        seen["hook_url"] = hook_url
        seen["event"] = event
        return {"action": "continue"}

    monkeypatch.setattr(mitm, "_send_hook_event", fake_send)

    mitm.tcp_message(flow)

    assert seen["hook_url"] == "http://127.0.0.1:4567/hook"
    event = seen["event"]
    assert event["kind"] == "tcp_message"
    assert event["protocol"] == "tcp"
    assert event["hook"] == "message"
    assert event["client"] == "('10.0.0.2', 5555)"
    assert event["server"] == "('10.0.0.3', 443)"
    assert base64.b64decode(event["message"]["body_base64"]) == b"\x00abc"


@pytest.mark.asyncio
async def test_openai_forwarder_translates_on_host(monkeypatch) -> None:
    monkeypatch.setenv("ABRIDGE_TRACE", "0")
    forwarder = mitm.OpenAIForwarder(
        base_url="http://127.0.0.1:8000/v1",
        api_key="sk-test",
        model="qwen",
        extra_body={"enable_thinking": True},
    )
    seen: dict[str, Any] = {}

    async def fake_post(body: dict[str, Any]) -> dict[str, Any]:
        seen.update(body)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "def add(a, b): return a + b"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        }

    monkeypatch.setattr(forwarder, "_post_openai", fake_post)

    action = await forwarder.handle_event(
        {
            "kind": "http_request",
            "protocol": "http",
            "hook": "request",
            "request": {
                "path": "/v1/messages",
                "json": {
                    "model": "claude",
                    "system": "be terse",
                    "stream": True,
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                    "tools": [
                        {
                            "name": "bash",
                            "description": "run command",
                            "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                        }
                    ],
                },
            }
        }
    )

    assert action is not None
    assert action["action"] == "respond"
    envelope = action["response"]
    assert seen["model"] == "qwen"
    assert seen["stream"] is False
    assert seen["enable_thinking"] is True
    assert seen["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]
    assert seen["tools"][0]["function"]["name"] == "bash"
    assert envelope["status_code"] == 200
    assert envelope["headers"]["content-type"] == "text/event-stream"
    content = base64.b64decode(envelope["body_base64"])
    assert b"def add(a, b): return a + b" in content


def test_mitmdump_args_default_to_wireguard_with_this_addon(monkeypatch) -> None:
    monkeypatch.delenv("ABRIDGE_MITM_MODE", raising=False)

    args = mitm._mitmdump_args(["--listen-port", "9090"])

    assert args[:4] == ["-s", str(mitm.ADDON_SCRIPT), "--mode", "wireguard"]
    assert args[4:] == ["--listen-port", "9090"]


def test_mitmdump_args_keep_user_mode_and_script(monkeypatch) -> None:
    monkeypatch.setenv("ABRIDGE_MITM_MODE", "wireguard")

    args = mitm._mitmdump_args(["--mode", "regular", "-s", "custom.py"])

    assert args == ["--mode", "regular", "-s", "custom.py"]


def test_mitmdump_args_pass_through_help() -> None:
    assert mitm._mitmdump_args(["--help"]) == ["--help"]


def test_response_sends_http_response_proxy_event(monkeypatch) -> None:
    monkeypatch.setenv("ABRIDGE_TRACE", "0")
    monkeypatch.setenv("ABRIDGE_HOOK_URL", "http://127.0.0.1:4567/hook")
    flow = _Flow({"model": "claude", "messages": [{"role": "user", "content": "hi"}]})
    flow.response = _Response({"ok": True}, status_code=201)
    seen: dict[str, Any] = {}

    def fake_send(event: dict[str, Any], *, hook_url: str) -> dict[str, Any]:
        seen["hook_url"] = hook_url
        seen["event"] = event
        return {"action": "continue"}

    monkeypatch.setattr(mitm, "_send_hook_event", fake_send)

    mitm.response(flow)

    assert seen["hook_url"] == "http://127.0.0.1:4567/hook"
    event = seen["event"]
    assert event["kind"] == "http_response"
    assert event["protocol"] == "http"
    assert event["hook"] == "response"
    assert event["response"]["status_code"] == 201
    assert json.loads(base64.b64decode(event["response"]["body_base64"])) == {"ok": True}
