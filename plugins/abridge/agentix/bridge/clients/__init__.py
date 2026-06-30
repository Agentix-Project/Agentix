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

All three classes expose `environ(handle)` (instance method) — the env-var
bundle an in-sandbox SDK needs to route through the tunnel, so the wiring step
is `env=client.environ(handle)` uniformly. `OpenAIClient` returns
`{OPENAI_BASE_URL: handle.url + "/v1", OPENAI_API_KEY: placeholder}` — the `/v1`
suffix the OpenAI SDK expects is baked in so a caller can't drop it; the two
Anthropic-side classes return `{ANTHROPIC_BASE_URL: handle.url, ANTHROPIC_API_KEY:
placeholder}` (no `/v1` — the Anthropic SDK appends it itself).

The two `populate_*_span` helpers are exposed at this level so user-
written clients can stamp the same OTel GenAI attrs the bundled
clients do.
"""

from __future__ import annotations

from ._genai_span import populate_anthropic_span, populate_openai_span
from .anthropic import PLACEHOLDER_API_KEY as ANTHROPIC_PLACEHOLDER_API_KEY
from .anthropic import AnthropicClient
from .anthropic_from_openai import AnthropicFromOpenAIClient
from .anthropic_to_openai import AnthropicToOpenAI
from .openai import PLACEHOLDER_API_KEY as OPENAI_PLACEHOLDER_API_KEY
from .openai import OpenAIClient

__all__ = [
    "ANTHROPIC_PLACEHOLDER_API_KEY",
    "AnthropicClient",
    "AnthropicFromOpenAIClient",
    "AnthropicToOpenAI",
    "OPENAI_PLACEHOLDER_API_KEY",
    "OpenAIClient",
    "populate_anthropic_span",
    "populate_openai_span",
]
