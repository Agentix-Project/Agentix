"""Tests for `agentix.bridge.detection`."""

from __future__ import annotations

import pytest
from agentix.bridge.detection import ApiFamily, detect


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/v1/messages", ApiFamily.ANTHROPIC_MESSAGES),
        ("/v1/messages?beta=1", ApiFamily.ANTHROPIC_MESSAGES),
        ("/v1/messages/", ApiFamily.ANTHROPIC_MESSAGES),
        ("/v1/messages/count_tokens", ApiFamily.ANTHROPIC_COUNT_TOKENS),
        ("/v1/chat/completions", ApiFamily.OPENAI_CHAT_COMPLETIONS),
        ("/openai/v1/chat/completions", ApiFamily.OPENAI_CHAT_COMPLETIONS),
        ("/chat/completions", ApiFamily.OPENAI_CHAT_COMPLETIONS),
        ("/v1/embeddings", ApiFamily.UNKNOWN),
        ("/", ApiFamily.UNKNOWN),
    ],
)
def test_detect_from_path(path: str, expected: ApiFamily) -> None:
    assert detect(path) is expected


def test_count_tokens_more_specific_than_messages() -> None:
    # Even though /v1/messages/count_tokens ends with neither of the
    # "messages" suffixes, the more-specific rule matches first.
    assert detect("/v1/messages/count_tokens") is ApiFamily.ANTHROPIC_COUNT_TOKENS


def test_body_fallback_anthropic_with_max_tokens() -> None:
    body = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    # path doesn't hint anything; body has anthropic-ish shape.
    assert detect("/proxy/inbound", body) is ApiFamily.ANTHROPIC_MESSAGES


def test_body_fallback_openai_without_max_tokens() -> None:
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    assert detect("/proxy/inbound", body) is ApiFamily.OPENAI_CHAT_COMPLETIONS


def test_family_classification_flags() -> None:
    assert ApiFamily.ANTHROPIC_MESSAGES.is_anthropic
    assert ApiFamily.ANTHROPIC_COUNT_TOKENS.is_anthropic
    assert not ApiFamily.ANTHROPIC_MESSAGES.is_openai
    assert ApiFamily.OPENAI_CHAT_COMPLETIONS.is_openai
    assert not ApiFamily.OPENAI_CHAT_COMPLETIONS.is_anthropic
    assert not ApiFamily.UNKNOWN.is_anthropic
    assert not ApiFamily.UNKNOWN.is_openai
