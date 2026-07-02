"""`OpenAIClient` — a standard OpenAI Chat Completions handler.

Uses the `openai` SDK (`openai.AsyncOpenAI`) to talk to any
OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, your gateway).
Registers one `@on("/v1/chat/completions")` handler — the same path
the OpenAI SDK on the agent side would hit when its `base_url` points
at our tunnel.

The SDK accepts the OpenAI Chat Completions request shape via typed
kwargs. Upstream failures surface as `AbridgeError` carrying the
upstream HTTP status (so the agent sees the real 429/400/... instead of
a blanket 502); for arbitrary-shape forwarding write your own
`@on("/v1/chat/completions")` handler with raw httpx.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from agentix.utils import trace

from ..proxy import AbridgeError, ClientResponse, Request, TunnelHandle, on
from ._genai_span import populate_openai_span

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAIError
else:
    # `openai` is an optional dep: install via `agentix-bridge[openai]`.
    # We try the import here so any module that lazily uses `OpenAIClient`
    # can still import this module; instantiation raises a helpful error
    # when the SDK is missing.
    try:
        from openai import AsyncOpenAI, OpenAIError
    except ImportError:
        AsyncOpenAI = None
        OpenAIError = Exception

logger = logging.getLogger(__name__)


_INSTALL_HINT = (
    "OpenAIClient requires the openai SDK. "
    "Install with: pip install 'agentix-bridge[openai]'"
)

# OpenAI SDK and many downstream tools check for the `sk-` prefix and
# a sensible length (~48 chars) before accepting the key. Our
# placeholder satisfies the format check and clearly identifies itself
# as a non-secret stand-in; the real upstream key lives only on the
# host (this client).
PLACEHOLDER_API_KEY = "sk-abridge-placeholder-no-real-credentials-leaked-into-sandbox"


class OpenAIClient:
    """POST any OpenAI Chat Completions body to an OpenAI-compatible
    upstream.

    `session_id` identifies the rollout this client serves; it stamps
    `x-session-id` on every upstream call so a gateway can group its
    own token-level trajectory. Auto-generated if not passed. Reusing
    the same `OpenAIClient` across multiple `Proxy` instances means
    they all share one session — that's the intended model.

    `model`, if set, overrides the agent's `model` field so many agents
    can share one upstream. `extra_body` is merged in last for custom
    fields the SDK doesn't model natively (reasoning effort, response
    format, …).
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120.0,
        session_id: str | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError(_INSTALL_HINT)
        self._sdk = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._model = model
        self._extra_body = extra_body or {}
        self.session_id = session_id or uuid.uuid4().hex

    @on("/v1/chat/completions")
    async def chat(self, request: Request) -> ClientResponse:
        body = dict(request.body)
        if self._model:
            body["model"] = self._model
        if self._extra_body:
            body.update(self._extra_body)
        record_id = uuid.uuid4().hex
        extra_headers = {
            "x-session-id": self.session_id,
            "x-request-id": record_id,
        }
        # Open the OTel span here — abridge's `Proxy` is trace-blind and
        # the caller's contextvars don't propagate across the HTTP/SIO
        # boundary, so this is the only point where a span is reliably
        # in scope for `populate_openai_span` to find.
        with trace.span(f"openai chat {body.get('model') or ''}"):
            try:
                completion = await self._sdk.chat.completions.create(
                    **body, extra_headers=extra_headers
                )
            except OpenAIError as exc:
                status = int(getattr(exc, "status_code", 502) or 502)
                raise AbridgeError(f"openai: {exc}", status_code=status) from exc
            response_dict = completion.model_dump(exclude_none=False)
            populate_openai_span(request=request.body, response=response_dict)
            return ClientResponse.json(response_dict)

    def environ(self, handle: TunnelHandle) -> dict[str, str]:
        """Env-var bundle an in-sandbox OpenAI SDK needs to route through `handle`,
        mirroring `AnthropicClient.environ`. `OPENAI_BASE_URL` carries the `/v1`
        suffix the OpenAI SDK expects (it appends only `/chat/completions`), so the
        load-bearing suffix is baked in here instead of left to the caller.
        `OPENAI_API_KEY` is a non-secret placeholder whose shape passes the SDK's
        local format check — the real upstream key lives on the host (this client)."""
        return {
            "OPENAI_BASE_URL": handle.url + "/v1",
            "OPENAI_API_KEY": PLACEHOLDER_API_KEY,
        }


__all__ = ["PLACEHOLDER_API_KEY", "OpenAIClient"]
