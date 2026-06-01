"""Sandbox tunnel + host `Proxy` round-trip.

Hits the in-process `wired` harness (see conftest) with an
Anthropic-format request and an OpenAI-format request, then with the
streaming and count-tokens edge cases. Each call exercises the full
path-named SIO event handshake and the bundled
`AnthropicFromOpenAIClient` / `OpenAIClient` handlers."""

from __future__ import annotations

import httpx
import pytest
from agentix.bridge import TunnelHandle
from agentix.bridge.clients import AnthropicClient
from agentix.bridge.clients.anthropic import PLACEHOLDER_API_KEY as ANTHROPIC_PLACEHOLDER_API_KEY


@pytest.mark.asyncio
async def test_anthropic_request_routes_through_tunnel(wired) -> None:
    handle = wired["handle"]
    upstream = wired["upstream"]

    async with httpx.AsyncClient(base_url=handle.url, timeout=10) as c:
        r = await c.post(
            "/v1/messages",
            json={
                "model": "claude-3-haiku",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "claude-3-haiku"  # response carries the agent's model name
    assert body["content"][0]["text"] == "hello from upstream"
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 4}

    # Upstream saw the *translated* body with the override model...
    assert upstream.last_body["model"] == "upstream-model"
    assert upstream.last_body["messages"][-1]["content"] == "hi"
    # ...and the rollout identity forwarded as a header.
    assert upstream.last_headers.get("x-session-id") == "sess-test"


@pytest.mark.asyncio
async def test_openai_request_routes_through_tunnel(wired) -> None:
    handle = wired["handle"]

    async with httpx.AsyncClient(base_url=handle.url, timeout=10) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello from upstream"


@pytest.mark.asyncio
async def test_anthropic_streaming_returns_sse(wired) -> None:
    handle = wired["handle"]

    async with httpx.AsyncClient(base_url=handle.url, timeout=10) as c:
        r = await c.post(
            "/v1/messages",
            json={
                "model": "claude-3-haiku",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"event: message_start" in r.content
    assert b"event: message_stop" in r.content


@pytest.mark.asyncio
async def test_count_tokens_handled_in_client_without_upstream(wired) -> None:
    """`AnthropicFromOpenAI.count_tokens` resolves locally, no upstream call."""
    handle = wired["handle"]
    upstream = wired["upstream"]
    upstream.last_body = {}  # reset

    async with httpx.AsyncClient(base_url=handle.url, timeout=10) as c:
        r = await c.post(
            "/v1/messages/count_tokens",
            json={"messages": [{"role": "user", "content": "y" * 16}]},
        )
    assert r.status_code == 200
    assert r.json() == {"input_tokens": 4}
    assert upstream.last_body == {}


@pytest.mark.asyncio
async def test_unregistered_path_returns_404(wired) -> None:
    """The sandbox tunnel only installs routes for the declared paths;
    everything else 404s. That's the @on whitelist in action."""
    handle = wired["handle"]
    async with httpx.AsyncClient(base_url=handle.url, timeout=10) as c:
        r = await c.post("/some/custom/path", json={"x": 1})
    assert r.status_code == 404


def test_anthropic_client_environ_shape() -> None:
    """`AnthropicClient.environ(handle)` returns the two env vars an
    Anthropic SDK needs to route through the tunnel. The placeholder
    `ANTHROPIC_API_KEY` matches Anthropic's real key format
    (`sk-ant-api03-...`) so the SDK's local validation accepts it,
    while the body of the key makes clear it's not a real credential."""
    client = AnthropicClient(api_key="real-upstream-key")
    handle = TunnelHandle(url="http://127.0.0.1:8000", port=8000)
    env = client.environ(handle)
    assert env == {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
        "ANTHROPIC_API_KEY": ANTHROPIC_PLACEHOLDER_API_KEY,
    }
    assert env["ANTHROPIC_API_KEY"].startswith("sk-ant-api03-")
