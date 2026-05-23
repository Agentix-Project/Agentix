"""Request/response transformers between LLM API families.

Today every captured request lands at the host as an OpenAI Chat
Completions body — that is the only client the host ships. The
transformers here are the layer that gets us there, and back, for
each non-OpenAI family the proxy recognises.

Layout mirrors `polar.gateway.transform/`:

  * one module per source family (`anthropic.py`)
  * each module exposes a small, typed surface
    (`request_to_openai(body)` / `response_to_native(...)`)

When we later add the Gemini, Cohere, or Bedrock families, they get
their own modules here.
"""

from __future__ import annotations

from .anthropic import (
    AnthropicCountTokens,
    anthropic_messages_to_openai,
    anthropic_sse,
    count_anthropic_tokens,
    openai_to_anthropic_messages,
)

__all__ = [
    "AnthropicCountTokens",
    "anthropic_messages_to_openai",
    "anthropic_sse",
    "count_anthropic_tokens",
    "openai_to_anthropic_messages",
]
