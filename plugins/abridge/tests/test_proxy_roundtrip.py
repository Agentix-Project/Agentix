"""Sandbox proxy + host client round-trip.

Issues an Anthropic-format and an OpenAI-format request through the
in-process `wired` harness (see conftest) and asserts they round-trip,
forward the rollout identity upstream, and produce session-grouped
`CompletionRecord`s with correct usage.
"""

from __future__ import annotations

import asyncio

import agentix.bridge.proxy as proxy_mod
import httpx
import pytest
from agentix.bridge import InMemoryStore, detect
from agentix.bridge.detection import ApiFamily


@pytest.mark.asyncio
async def test_anthropic_request_routes_through_proxy(wired) -> None:
    handle = wired["handle"]
    store: InMemoryStore = wired["store"]
    upstream = wired["upstream"]

    async with httpx.AsyncClient(base_url=handle.anthropic_base_url, timeout=10) as c:
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

    # Upstream saw the *translated* body with our overridden model...
    assert upstream.last_body["model"] == "upstream-model"
    assert upstream.last_body["messages"][-1]["content"] == "hi"
    # ...and the rollout identity forwarded as a header for the gateway.
    assert upstream.last_headers.get("x-session-id") == "sess-test"

    # A CompletionRecord landed in the host store, grouped by session.
    await asyncio.sleep(0.05)  # let the fire-and-forget emit drain
    records = store.snapshot()
    assert len(records) == 1
    rec = records[0]
    assert rec.family is ApiFamily.ANTHROPIC_MESSAGES
    assert rec.usage.prompt_tokens == 7
    assert rec.usage.completion_tokens == 4
    assert rec.session_id == "sess-test"
    assert store.trajectory("sess-test") == records


@pytest.mark.asyncio
async def test_openai_request_routes_through_proxy(wired) -> None:
    handle = wired["handle"]
    store: InMemoryStore = wired["store"]

    async with httpx.AsyncClient(base_url=handle.openai_base_url, timeout=10) as c:
        r = await c.post(
            "/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello from upstream"

    await asyncio.sleep(0.05)
    records = store.snapshot()
    assert len(records) == 1
    assert records[0].family is ApiFamily.OPENAI_CHAT_COMPLETIONS


def test_export_environ_shape() -> None:
    handle = proxy_mod.ProxyHandle(
        proxy_id="x",
        url="http://127.0.0.1:8000",
        port=8000,
        anthropic_base_url="http://127.0.0.1:8000",
        openai_base_url="http://127.0.0.1:8000/v1",
    )
    env = proxy_mod.export_environ(handle)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:8000/v1"
    # Placeholder keys only — never a real key read from the environment.
    assert env["ANTHROPIC_API_KEY"] == "sk-abridge"
    assert env["OPENAI_API_KEY"] == "sk-abridge"


def test_detect_anthropic_count_tokens_path() -> None:
    assert detect("/v1/messages/count_tokens") is ApiFamily.ANTHROPIC_COUNT_TOKENS
