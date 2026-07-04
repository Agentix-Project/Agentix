"""`AnthropicClient` — pure Anthropic Messages passthrough.

Uses the `anthropic` SDK (`anthropic.AsyncAnthropic`) to talk to any
Anthropic-protocol endpoint (api.anthropic.com or anything that speaks
the same shape). Registers two `@on(...)` handlers:

  * `/v1/messages` — Messages create
  * `/v1/messages/count_tokens` — token counting

Streaming-vs-not is honoured (`body["stream"]`): for `stream=True` the
SDK's streaming primitive runs and the chunks are re-serialised as the
Anthropic SSE wire envelope. For `stream=False`, a single JSON response
goes back.

`environ(handle)` returns the env-var bundle (`ANTHROPIC_BASE_URL` +
placeholder `ANTHROPIC_API_KEY`) an in-sandbox Anthropic SDK needs to
route through the tunnel. Instance method, not module-level — the env
vars an agent needs are tied to a (client, handle) pair, and methods
keep that relationship explicit.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from agentix.utils import trace

from ..proxy import (
    AbridgeError,
    ClientResponse,
    Request,
    TunnelHandle,
    on,
)
from ._genai_span import populate_anthropic_span

if TYPE_CHECKING:
    from anthropic import AnthropicError, AsyncAnthropic
else:
    # `anthropic` is an optional dep: install via `agentix-bridge[anthropic]`.
    try:
        from anthropic import AnthropicError, AsyncAnthropic
    except ImportError:
        AsyncAnthropic = None
        AnthropicError = Exception

logger = logging.getLogger(__name__)


_INSTALL_HINT = (
    "AnthropicClient requires the anthropic SDK. "
    "Install with: pip install 'agentix-bridge[anthropic]'"
)

# Anthropic SDKs / Claude Code validate the key's shape locally before
# any HTTP call — real keys are `sk-ant-api03-<~95 chars>`. Our
# placeholder matches that prefix so the SDK accepts it; the body of
# the key spells out that it's not a credential, so leak-detectors and
# log scrapers don't flag it as a secret. The real upstream key lives
# only on the host (this client).
PLACEHOLDER_API_KEY = (
    "sk-ant-api03-abridge-placeholder-no-real-credentials-leaked-into-sandbox"
)


class AnthropicClient:
    """Forward Anthropic Messages requests verbatim to a native
    Anthropic endpoint. No shape translation; the upstream IS Anthropic.

    `session_id` identifies the rollout this client serves; stamped as
    `x-session-id` on every upstream call (auto-gen if not passed).
    Sharing the same `AnthropicClient` across multiple `Proxy` instances
    means they all map to one session — the intended model.

    `model`, if set, overrides the agent's `model` field. Useful for
    pinning many agents to one upstream model without rewriting agents.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 0,
        session_id: str | None = None,
    ) -> None:
        if AsyncAnthropic is None:
            raise ImportError(_INSTALL_HINT)
        # 0, not the SDK's 2: silent retries multiply tunnel-window occupancy;
        # retry policy is the operator's call (see Proxy.request_timeout).
        self._client = AsyncAnthropic(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self._model = model
        self.session_id = session_id or uuid.uuid4().hex

    @on("/v1/messages")
    async def messages(self, request: Request) -> ClientResponse:
        body = dict(request.body)
        if self._model:
            body["model"] = self._model
        record_id = uuid.uuid4().hex
        extra_headers = {
            "x-session-id": self.session_id,
            "x-request-id": record_id,
        }
        stream = bool(body.pop("stream", False))

        # Open OTel span here (Proxy is trace-blind; caller's contextvar
        # doesn't propagate across HTTP/SIO).
        with trace.span(f"anthropic messages {body.get('model') or ''}"):
            try:
                if stream:
                    # The SDK exposes a streaming primitive; we drain it
                    # and synthesise the original Anthropic SSE blob so
                    # the agent's SDK sees a wire stream regardless of
                    # how the SDK chose to read events.
                    async with self._client.messages.stream(
                        **body, extra_headers=extra_headers
                    ) as stream_handle:
                        async for _ in stream_handle:
                            pass
                        message = await stream_handle.get_final_message()
                    response_dict = message.model_dump(exclude_none=False)
                    populate_anthropic_span(request=request.body, response=response_dict)
                    return ClientResponse.sse(_anthropic_sse_from_message(response_dict))

                message = await self._client.messages.create(
                    **body, extra_headers=extra_headers
                )
            except AnthropicError as exc:
                status = int(getattr(exc, "status_code", 502) or 502)
                raise AbridgeError(f"anthropic: {exc}", status_code=status) from exc
            response_dict = message.model_dump(exclude_none=False)
            populate_anthropic_span(request=request.body, response=response_dict)
            return ClientResponse.json(response_dict)

    @on("/v1/messages/count_tokens")
    async def count_tokens(self, request: Request) -> ClientResponse:
        body = dict(request.body)
        if self._model:
            body["model"] = self._model
        try:
            result = await self._client.messages.count_tokens(**body)
        except AnthropicError as exc:
            status = int(getattr(exc, "status_code", 502) or 502)
            raise AbridgeError(f"anthropic: {exc}", status_code=status) from exc
        return ClientResponse.json(result.model_dump(exclude_none=False))

    def environ(self, handle: TunnelHandle) -> dict[str, str]:
        """Env-var bundle an in-sandbox Anthropic SDK needs to route
        through `handle`. `ANTHROPIC_API_KEY` is a non-secret placeholder
        whose shape matches Anthropic's real key format so the SDK's
        local validation passes — the real upstream credentials live on
        the host (this client)."""
        return {
            "ANTHROPIC_BASE_URL": handle.url,
            "ANTHROPIC_API_KEY": PLACEHOLDER_API_KEY,
        }


def _anthropic_sse_from_message(message: dict[str, Any]) -> bytes:
    """Re-serialise a completed Anthropic Message as a single-shot SSE
    blob. Imported lazily to avoid pulling the OpenAI-translation
    helpers when only the pure Anthropic client is used."""
    from ._anthropic_transforms import anthropic_sse  # noqa: PLC0415

    return anthropic_sse(message)


__all__ = ["PLACEHOLDER_API_KEY", "AnthropicClient"]
