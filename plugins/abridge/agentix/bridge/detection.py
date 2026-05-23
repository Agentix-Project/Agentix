"""Detect which LLM API family a captured HTTP request belongs to.

Mirrors `polar.gateway.detection`: the gateway needs to know which
API family a request is targeting so the transform layer can pick the
right converter. Currently we recognise:

  * Anthropic Messages (`POST /v1/messages`)
  * Anthropic Count-Tokens (`POST /v1/messages/count_tokens`)
  * OpenAI Chat Completions (`POST /v1/chat/completions`,
    `POST /chat/completions`, and provider-prefixed variants like
    `POST /openai/v1/chat/completions`)

Detection is structural — we look at the URL path and (when ambiguous)
the request body. We deliberately do NOT trust an upstream `Host`
header because the proxy listens on `127.0.0.1`; the agent SDK simply
reroutes its base URL through us.
"""

from __future__ import annotations

import enum
from typing import Any


class ApiFamily(enum.StrEnum):
    """Which API family a captured request belongs to."""

    ANTHROPIC_MESSAGES = "anthropic.messages"
    ANTHROPIC_COUNT_TOKENS = "anthropic.count_tokens"
    OPENAI_CHAT_COMPLETIONS = "openai.chat.completions"
    UNKNOWN = "unknown"

    @property
    def is_anthropic(self) -> bool:
        return self in (
            ApiFamily.ANTHROPIC_MESSAGES,
            ApiFamily.ANTHROPIC_COUNT_TOKENS,
        )

    @property
    def is_openai(self) -> bool:
        return self == ApiFamily.OPENAI_CHAT_COMPLETIONS


_ANTHROPIC_MESSAGES_SUFFIXES = ("/v1/messages",)
_ANTHROPIC_COUNT_TOKENS_SUFFIXES = ("/v1/messages/count_tokens",)
_OPENAI_CHAT_SUFFIXES = (
    "/v1/chat/completions",
    "/chat/completions",
)


def _path_only(path: str) -> str:
    return path.split("?", 1)[0].rstrip("/") or "/"


def detect(path: str, body: dict[str, Any] | None = None) -> ApiFamily:
    """Classify a captured request from its URL path + JSON body."""
    p = _path_only(path)

    # Count-tokens is more specific than messages; check it first.
    for suffix in _ANTHROPIC_COUNT_TOKENS_SUFFIXES:
        if p.endswith(suffix):
            return ApiFamily.ANTHROPIC_COUNT_TOKENS

    for suffix in _ANTHROPIC_MESSAGES_SUFFIXES:
        if p.endswith(suffix):
            return ApiFamily.ANTHROPIC_MESSAGES

    for suffix in _OPENAI_CHAT_SUFFIXES:
        if p.endswith(suffix):
            return ApiFamily.OPENAI_CHAT_COMPLETIONS

    # Body-level fallback when the path is unusual but the shape is
    # recognisable (e.g. a custom-prefixed proxy in front of us).
    if isinstance(body, dict):
        if "messages" in body and "max_tokens" in body and "model" in body:
            # Anthropic Messages always has `max_tokens`; OpenAI does
            # not strictly require it.
            return ApiFamily.ANTHROPIC_MESSAGES
        if "messages" in body and "model" in body:
            return ApiFamily.OPENAI_CHAT_COMPLETIONS

    return ApiFamily.UNKNOWN


__all__ = ["ApiFamily", "detect"]
