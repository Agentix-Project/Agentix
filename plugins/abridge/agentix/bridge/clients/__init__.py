"""Bundled handler clients for abridge.

Three out-of-the-box implementations, all built around the official
provider SDKs (`openai`, `anthropic`). Each is a plain class with
`@on(path)`-decorated methods — pass an instance to `Proxy(...)` or
mixin-compose multiple in one user-defined class.

  * `OpenAIClient` — agent speaks OpenAI Chat Completions, upstream is
    OpenAI-compatible. One `@on("/v1/chat/completions")`.
  * `AnthropicClient` — agent speaks Anthropic Messages, upstream is
    native Anthropic. `@on("/v1/messages")` + `@on("/v1/messages/count_tokens")`.
  * `AnthropicFromOpenAIClient` — agent speaks Anthropic, upstream is
    OpenAI-compatible (translation lives here). Same path set as
    `AnthropicClient`.

The two Anthropic-side classes also expose `environ(handle)` (instance
method) — the env-var bundle (`ANTHROPIC_BASE_URL` + placeholder
`ANTHROPIC_API_KEY`) an in-sandbox Anthropic SDK needs to route through
the tunnel. The OpenAI client doesn't ship an `environ`; agents that
use the OpenAI SDK typically construct the client with
`base_url=handle.url + "/v1"` directly.

The two `populate_*_span` helpers are exposed at this level so user-
written clients can stamp the same OTel GenAI attrs the bundled
clients do.
"""

from __future__ import annotations

from ._genai_span import populate_anthropic_span, populate_openai_span
from .anthropic import PLACEHOLDER_API_KEY as ANTHROPIC_PLACEHOLDER_API_KEY
from .anthropic import AnthropicClient
from .anthropic_from_openai import AnthropicFromOpenAIClient
from .openai import PLACEHOLDER_API_KEY as OPENAI_PLACEHOLDER_API_KEY
from .openai import OpenAIClient

__all__ = [
    "ANTHROPIC_PLACEHOLDER_API_KEY",
    "AnthropicClient",
    "AnthropicFromOpenAIClient",
    "OPENAI_PLACEHOLDER_API_KEY",
    "OpenAIClient",
    "populate_anthropic_span",
    "populate_openai_span",
]
