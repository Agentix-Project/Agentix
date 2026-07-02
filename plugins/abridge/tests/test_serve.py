"""Tests for `agentix.bridge.serve` — the tunnel-less direct mode.

The HTTP contract must match the sandbox tunnel (JSON-object bodies,
`ClientResponse` out, in-band errors), and the session-keyed app must
map caller keys to stable, distinct, LRU-managed client sessions.
"""

from __future__ import annotations

from typing import Any

from agentix.bridge import AbridgeError, ClientResponse, Request, on
from agentix.bridge.serve import _session_id, build_app, build_session_app
from fastapi.testclient import TestClient


class EchoClient:
    def __init__(self, session_id: str = "fixed") -> None:
        self.session_id = session_id
        self.closed = False

    @on("/v1/echo")
    async def echo(self, request: Request) -> ClientResponse:
        return ClientResponse.json({"echo": request.body, "session": self.session_id})

    @on("/v1/teapot")
    async def teapot(self, request: Request) -> ClientResponse:
        raise AbridgeError("short and stout", status_code=418)

    @on("/v1/boom")
    async def boom(self, request: Request) -> ClientResponse:
        raise RuntimeError("kaput")

    async def aclose(self) -> None:
        self.closed = True


def test_build_app_serves_handlers_and_health() -> None:
    tc = TestClient(build_app(EchoClient()))
    assert tc.get("/_health").json() == {"status": "ok"}
    r = tc.post("/v1/echo", json={"x": 1})
    assert r.status_code == 200
    assert r.json()["echo"] == {"x": 1}
    assert tc.post("/nope", json={}).status_code == 404


def test_handler_errors_become_wire_errors() -> None:
    tc = TestClient(build_app(EchoClient()))
    r = tc.post("/v1/teapot", json={})
    assert r.status_code == 418
    assert "short and stout" in r.json()["error"]["message"]
    r = tc.post("/v1/boom", json={})
    assert r.status_code == 502
    assert "RuntimeError" in r.json()["error"]["message"]


def test_non_object_body_coerced_to_empty_like_the_tunnel() -> None:
    tc = TestClient(build_app(EchoClient()))
    r = tc.post("/v1/echo", content=b"[1, 2]", headers={"content-type": "application/json"})
    assert r.json()["echo"] == {}


def test_session_app_maps_keys_to_stable_distinct_sessions() -> None:
    tc = TestClient(build_session_app(EchoClient))
    a1 = tc.post("/v1/echo", json={}, headers={"x-api-key": "sk-a"}).json()["session"]
    a2 = tc.post("/v1/echo", json={}, headers={"x-api-key": "sk-a"}).json()["session"]
    bearer = tc.post("/v1/echo", json={}, headers={"authorization": "Bearer sk-b"}).json()["session"]
    anonymous = tc.post("/v1/echo", json={}).json()["session"]

    assert a1 == a2 == _session_id("sk-a")
    assert bearer == _session_id("sk-b")
    assert bearer != a1
    assert anonymous == "anonymous"


def test_session_app_evicts_and_closes_least_recent() -> None:
    built: list[EchoClient] = []

    def factory(session_id: str) -> EchoClient:
        client = EchoClient(session_id)
        built.append(client)
        return client

    tc = TestClient(build_session_app(factory, max_sessions=1))
    tc.post("/v1/echo", json={}, headers={"x-api-key": "sk-a"})
    tc.post("/v1/echo", json={}, headers={"x-api-key": "sk-b"})

    still_open = [client.session_id for client in built if not client.closed]
    assert still_open == [_session_id("sk-b")]
    assert [client.session_id for client in built if client.closed] == ["anonymous", _session_id("sk-a")]


def _mock_completion() -> Any:
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "upstream-model",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }
    )


def test_agent_keys_become_upstream_session_headers() -> None:
    """End to end through the real translation client: the key each
    agent sends is hashed into the `x-session-id` the upstream sees, and
    the agent's key itself never reaches the upstream call."""
    from agentix.bridge.clients import AnthropicFromOpenAIClient

    seen: list[dict[str, str]] = []

    def factory(session_id: str) -> AnthropicFromOpenAIClient:
        client = AnthropicFromOpenAIClient(api_key="real-upstream-key", model="m", session_id=session_id)

        async def create(**kwargs: Any) -> Any:
            seen.append(dict(kwargs["extra_headers"]))
            return _mock_completion()

        client._client.chat.completions.create = create  # type: ignore[method-assign]
        return client

    tc = TestClient(build_session_app(factory))
    body = {"model": "claude", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    assert tc.post("/v1/messages", json=body, headers={"x-api-key": "rollout-1"}).status_code == 200
    assert tc.post("/v1/messages", json=body, headers={"x-api-key": "rollout-2"}).status_code == 200

    assert seen[0]["x-session-id"] == _session_id("rollout-1")
    assert seen[1]["x-session-id"] == _session_id("rollout-2")
    assert seen[0]["x-session-id"] != seen[1]["x-session-id"]
    assert all("rollout-1" not in v and "rollout-2" not in v for headers in seen for v in headers.values())
