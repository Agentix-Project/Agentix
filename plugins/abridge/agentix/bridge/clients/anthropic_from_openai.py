"""`AnthropicFromOpenAIClient` — Anthropic Messages agent, OpenAI upstream.

The agent talks Anthropic (Claude Code, Anthropic SDK callers). The
upstream is OpenAI-compatible (OpenAI, OpenRouter, vLLM, your gateway).
This client translates the request body Anthropic → OpenAI, dispatches
via the `openai` SDK, and translates the response back to Anthropic
shape so the agent never knows the difference. SSE-renders the
response when the agent asked for streaming.

`environ(handle)` returns the same env-var bundle as the pure
`AnthropicClient` — from the agent's side the wire is identical.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from agentix.utils import trace

from ..proxy import (
    AbridgeError,
    ClientResponse,
    Request,
    TunnelHandle,
    on,
)
from ._anthropic_transforms import (
    anthropic_messages_to_openai,
    anthropic_sse,
    count_anthropic_tokens,
    openai_to_anthropic_messages,
)
from ._genai_span import populate_anthropic_span
from .anthropic import PLACEHOLDER_API_KEY

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAIError
else:
    # Depends on the OpenAI SDK (not anthropic) — the agent speaks
    # Anthropic but the upstream is OpenAI-compatible. Install via
    # `agentix-bridge[openai]`.
    try:
        from openai import AsyncOpenAI, OpenAIError
    except ImportError:
        AsyncOpenAI = None
        OpenAIError = Exception

logger = logging.getLogger(__name__)


_INSTALL_HINT = (
    "AnthropicFromOpenAIClient requires the openai SDK. "
    "Install with: pip install 'agentix-bridge[openai]'"
)


class AnthropicFromOpenAIClient:
    """Anthropic Messages → OpenAI Chat Completions adapter.

    `session_id` is auto-generated if not passed; stamped as
    `x-session-id` on every upstream call. Sharing one client across
    multiple `Proxy` instances means they share the session.

    `model`, when set, overrides the agent's `model` field in
    the OpenAI body — so the agent can keep sending its preferred
    Anthropic model id while the upstream gets whatever model name it
    actually serves.

    `count_tokens` is answered locally with a character-based estimate
    (no upstream call); the real Anthropic count_tokens API is
    tokenizer-specific and an OpenAI endpoint can't satisfy it.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        session_id: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError(_INSTALL_HINT)
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self.session_id = session_id or uuid.uuid4().hex

    @on("/v1/messages")
    async def messages(self, request: Request) -> ClientResponse:
        openai_body = anthropic_messages_to_openai(
            request.body, upstream_model=self._model
        )
        openai_body["stream"] = False
        record_id = uuid.uuid4().hex
        extra_headers = {
            "x-session-id": self.session_id,
            "x-request-id": record_id,
        }
        # Span named in the agent's POV (anthropic) — the upstream OpenAI
        # hop is an implementation detail. Proxy is trace-blind so this
        # is the right scope for `populate_anthropic_span` to find.
        with trace.span(f"anthropic messages {request.body.get('model') or ''}"):
            try:
                completion = await self._client.chat.completions.create(
                    **openai_body, extra_headers=extra_headers
                )
            except OpenAIError as exc:
                status = int(getattr(exc, "status_code", 502) or 502)
                raise AbridgeError(f"openai: {exc}", status_code=status) from exc

            openai_resp = completion.model_dump(exclude_none=False)
            anthropic_resp = openai_to_anthropic_messages(
                openai_resp, response_model=str(request.body.get("model") or "")
            )
            populate_anthropic_span(request=request.body, response=anthropic_resp)
            if request.body.get("stream"):
                return ClientResponse.sse(anthropic_sse(anthropic_resp))
            return ClientResponse.json(anthropic_resp)

    @on("/v1/messages/count_tokens")
    async def count_tokens(self, request: Request) -> ClientResponse:
        return ClientResponse.json(
            {"input_tokens": count_anthropic_tokens(request.body).input_tokens}
        )

    def environ(self, handle: TunnelHandle) -> dict[str, str]:
        """Same env vars as `AnthropicClient.environ` — from the agent's
        POV the wire is Anthropic regardless of the upstream's shape."""
        return {
            "ANTHROPIC_BASE_URL": handle.url,
            "ANTHROPIC_API_KEY": PLACEHOLDER_API_KEY,
        }


__all__ = ["AnthropicFromOpenAIClient"]
