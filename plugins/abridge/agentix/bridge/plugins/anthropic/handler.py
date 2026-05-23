"""Anthropic → OpenAI host handler."""

from __future__ import annotations

from typing import Any

from agentix.bridge.actions import continue_action, respond_action
from agentix.bridge.host.forwarder import HookForwarder
from agentix.bridge.plugins.anthropic.adapter_openai import forward_anthropic_http_event_async
from agentix.bridge.plugins.openai.upstream import post_openai_compatible
from agentix.bridge.types import ProxyAction, ProxyEvent, ResponseEnvelope


class AnthropicOpenAIHandler:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._extra_body = extra_body or {}
        self._timeout = timeout

    async def handle(self, event: ProxyEvent) -> ProxyAction | None:
        if event.get("kind") != "http_request":
            return None
        request = event.get("request") or {}
        if not isinstance(request, dict):
            return None
        path = str(request.get("path") or "").split("?", 1)[0]
        if path not in {"/v1/messages", "/v1/messages/count_tokens"}:
            return None
        envelope = await forward_anthropic_http_event_async(
            event,
            upstream_model=self._model,
            extra_body=self._extra_body,
            post_openai=self._post_openai,
        )
        return respond_action(envelope)

    async def _post_openai(self, body: dict[str, Any]) -> dict[str, Any]:
        return await post_openai_compatible(
            body,
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )


class OpenAIForwarder(HookForwarder):
    """Host-side SIO handler for sandbox mitmproxy flows."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._extra_body = extra_body or {}
        self._timeout = timeout

    async def handle_event(self, event: ProxyEvent) -> ProxyAction | None:
        if event.get("kind") != "http_request":
            return continue_action()
        request = event.get("request") or {}
        if not isinstance(request, dict):
            return continue_action()
        path = str(request.get("path") or "").split("?", 1)[0]
        if path not in {"/v1/messages", "/v1/messages/count_tokens"}:
            return continue_action()
        return respond_action(await self.forward(event))

    async def forward(self, event: ProxyEvent) -> ResponseEnvelope:
        return await forward_anthropic_http_event_async(
            event,
            upstream_model=self._model,
            extra_body=self._extra_body,
            post_openai=self._post_openai,
        )

    async def _post_openai(self, body: dict[str, Any]) -> dict[str, Any]:
        return await post_openai_compatible(
            body,
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
        )
