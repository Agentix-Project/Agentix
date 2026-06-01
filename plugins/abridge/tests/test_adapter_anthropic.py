"""Tests for `AnthropicFromOpenAIClient` — the Anthropic-side adapter.

The client uses the `openai` SDK to hit the upstream. Tests mock the
underlying SDK so they don't need a real OpenAI-compatible server, just
verify the Anthropic↔OpenAI translation + the `@on(path)` routing.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from agentix.bridge import ClientResponse, Request
from agentix.bridge.clients import AnthropicFromOpenAIClient


def _req(path: str, body: dict[str, Any]) -> Request:
    return Request(path=path, body=body)


def _mock_completion(content: str = "hi", prompt_tokens: int = 4, completion_tokens: int = 1) -> Any:
    """Build the shape the openai SDK returns from `chat.completions.create`."""
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "upstream-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


@pytest.mark.asyncio
async def test_anthropic_messages_translates_round_trip(monkeypatch) -> None:
    adapter = AnthropicFromOpenAIClient(api_key="k", model="gpt-4o")
    create = AsyncMock(return_value=_mock_completion(content="hi"))
    monkeypatch.setattr(adapter._client.chat.completions, "create", create)

    result = await adapter.messages(
        _req(
            "/v1/messages",
            {
                "model": "claude-3-haiku",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    )
    assert isinstance(result, ClientResponse)
    body = json.loads(result.body)
    assert body["model"] == "claude-3-haiku"  # response carries the agent's model name
    assert body["content"] == [{"type": "text", "text": "hi"}]
    assert body["usage"] == {"input_tokens": 4, "output_tokens": 1}

    # SDK received translated OpenAI body with override model + stream=False.
    args, kwargs = create.await_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["stream"] is False
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_anthropic_streaming_returns_sse(monkeypatch) -> None:
    adapter = AnthropicFromOpenAIClient(api_key="k")
    monkeypatch.setattr(
        adapter._client.chat.completions, "create",
        AsyncMock(return_value=_mock_completion()),
    )
    result = await adapter.messages(
        _req(
            "/v1/messages",
            {
                "model": "claude",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
    )
    assert isinstance(result, ClientResponse)
    assert result.media_type == "text/event-stream"
    assert b"event: message_start" in result.body
    assert b"event: message_stop" in result.body


@pytest.mark.asyncio
async def test_count_tokens_handled_locally_without_upstream(monkeypatch) -> None:
    adapter = AnthropicFromOpenAIClient(api_key="k")
    create = AsyncMock()
    monkeypatch.setattr(adapter._client.chat.completions, "create", create)

    result = await adapter.count_tokens(
        _req("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "y" * 16}]})
    )
    assert isinstance(result, ClientResponse)
    assert json.loads(result.body) == {"input_tokens": 4}
    create.assert_not_awaited()
