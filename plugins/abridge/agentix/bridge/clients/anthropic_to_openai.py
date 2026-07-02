"""`AnthropicToOpenAI` — the Convert capability: an Anthropic-Messages agent over
ANY OpenAI-compatible downstream.

This is one orthogonal capability — protocol translation — and nothing else. It
translates the agent's Anthropic `/v1/messages` to an OpenAI chat-completions body,
hands that body to a `downstream` (any abridge `Handler`), then translates the
OpenAI completion back to Anthropic. It is deliberately transport-blind: the
downstream owns the HTTP, the session, and any recording.

Compose it with a transport/routing capability:

    # plain OpenAI gateway (no session)
    Proxy(AnthropicToOpenAI(Forward(base_url, paths=["/v1/chat/completions"]).handler()))

    # session-scoped recorder (the TITO gateway) — session stays transparent
    tito  = SessionForward(gateway_url, paths=["/v1/chat/completions"])
    proxy = Proxy(AnthropicToOpenAI(tito.handler(), model="qwen3-4b"))
    ...                          # the agent only ever speaks Anthropic
    harvest(tito.session_id)     # the session lives in the SessionForward, not here

`AnthropicToOpenAI` knows nothing about sessions; `SessionForward` knows nothing
about Anthropic. They compose because the seam between them is just abridge's
`Handler` (an OpenAI chat body in, a `ClientResponse` out). No OpenAI SDK
dependency — the upstream hop is whatever `Handler` you pass.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentix.utils import trace

from ..proxy import ClientResponse, Handler, Request, TunnelHandle, _AsyncCloseable, on
from ._anthropic_transforms import (
    anthropic_messages_to_openai,
    anthropic_sse,
    count_anthropic_tokens,
    openai_to_anthropic_messages,
)
from ._genai_span import populate_anthropic_span
from .anthropic import PLACEHOLDER_API_KEY

logger = logging.getLogger(__name__)


class AnthropicToOpenAI:
    """Anthropic Messages agent → any OpenAI-compatible `downstream` Handler.

    `downstream` is the transport/routing capability this converter sits on top of
    — a `Forward(...).handler()` for a direct OpenAI gateway, or a
    `SessionForward(...).handler()` for a session-scoped recorder like the TITO
    gateway. `model`, when set, overrides the agent's model id in the OpenAI body.
    `count_tokens` is answered locally (character estimate, no downstream call).

    `aclose()` delegates to a closeable downstream (a `Forward.handler()` carries
    one), so `Proxy.stop` reaps the transport's HTTP pool through the composition.
    Everything else about the downstream stays yours: hold the forwarder to read
    `.session_id` / call `delete_session()`.
    """

    def __init__(self, downstream: Handler, *, model: str | None = None) -> None:
        self._downstream = downstream
        self._model = model
        # Memory of the EXACT assistant message the downstream returned, one
        # entry per tool-calling turn. The Anthropic round-trip is lossy (it
        # drops reasoning_content and the per-tool-call `index`), so a
        # reconstructed assistant won't match what a session-recording backend
        # (TITO) stored byte-for-byte. On the next turn we replay the remembered
        # original verbatim, keyed by the (id, name, canonical-args) triple of
        # each tool call — all of which survive the round-trip. Ids alone are
        # NOT a safe key: some servers reuse ids across turns (TGI emits "0"),
        # and an id-only key would replay the latest turn into every matching
        # history slot. Entries are never evicted — scope one converter
        # instance per rollout/conversation.
        self._assistant_by_calls: dict[
            tuple[tuple[str, str, str], ...], dict[str, Any]
        ] = {}

    @on("/v1/messages")
    async def messages(self, request: Request) -> ClientResponse:
        openai_body = anthropic_messages_to_openai(request.body, upstream_model=self._model)
        # The downstream produces a non-streaming OpenAI completion (a TITO
        # recorder needs output_token_logprobs); we re-render SSE locally below if
        # the agent asked for streaming.
        openai_body["stream"] = False
        self._replay_remembered_assistants(openai_body)
        with trace.span(f"anthropic messages {request.body.get('model') or ''}"):
            resp = await self._downstream(Request(path="/v1/messages", body=openai_body))
            if resp.status_code != 200:
                # Pass the downstream's error (4xx/5xx) straight through with its
                # status; the agent sees a non-200, not a malformed body.
                return resp
            openai_resp = json.loads(resp.body)
            self._remember_assistant(openai_resp)
            anthropic_resp = openai_to_anthropic_messages(
                openai_resp, response_model=str(request.body.get("model") or "")
            )
            populate_anthropic_span(request=request.body, response=anthropic_resp)
            if request.body.get("stream"):
                return ClientResponse.sse(anthropic_sse(anthropic_resp))
            return ClientResponse.json(anthropic_resp)

    @staticmethod
    def _call_key(message: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
        """One (id, name, canonical-args) triple per tool call. Arguments are
        re-serialized with sorted keys so the downstream's original JSON string
        and the round-trip's `json.dumps` rendering compare equal regardless of
        spacing or key order."""
        key: list[tuple[str, str, str]] = []
        for tc in message.get("tool_calls") or []:
            if not isinstance(tc, dict) or not tc.get("id"):
                continue
            function = tc.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.dumps(json.loads(raw_args), sort_keys=True)
            except ValueError:
                args = raw_args
            key.append((str(tc["id"]), str(function.get("name") or ""), args))
        return tuple(key)

    def _remember_assistant(self, openai_resp: dict[str, Any]) -> None:
        choice = (openai_resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = self._call_key(message)
        if calls:
            self._assistant_by_calls[calls] = message

    def _replay_remembered_assistants(self, openai_body: dict[str, Any]) -> None:
        messages = openai_body.get("messages")
        if not isinstance(messages, list):
            return
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            remembered = self._assistant_by_calls.get(self._call_key(msg))
            if remembered is not None:
                messages[i] = remembered

    @on("/v1/messages/count_tokens")
    async def count_tokens(self, request: Request) -> ClientResponse:
        return ClientResponse.json(
            {"input_tokens": count_anthropic_tokens(request.body).input_tokens}
        )

    async def aclose(self) -> None:
        """Close the downstream if it is closeable; a bare function is a no-op.

        Lets `Proxy.stop` reach the transport's HTTP pool through this
        converter — otherwise `Proxy(AnthropicToOpenAI(fwd.handler()))` leaks
        the forwarder's pool past the proxy lifecycle."""
        if isinstance(self._downstream, _AsyncCloseable):
            await self._downstream.aclose()

    def environ(self, handle: TunnelHandle) -> dict[str, str]:
        """Anthropic env-var bundle — from the agent's POV the wire is Anthropic,
        regardless of the OpenAI downstream."""
        return {
            "ANTHROPIC_BASE_URL": handle.url,
            "ANTHROPIC_API_KEY": PLACEHOLDER_API_KEY,
        }


__all__ = ["AnthropicToOpenAI"]
