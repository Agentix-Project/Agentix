"""Built-in sandbox interceptors."""

from __future__ import annotations

from agentix.bridge.actions import json_envelope, respond_action, response_envelope
from agentix.bridge.events import http_request_event
from agentix.bridge.mitm.hook_client import hook_url, send_hook_event
from agentix.bridge.pipeline.context import RequestContext
from agentix.bridge.plugins.anthropic.adapter_openai import (
    anthropic_client_envelope,
    anthropic_to_openai_body,
)
from agentix.bridge.plugins.anthropic.matchers import (
    blocked_host,
    is_anthropic_count_tokens,
    is_anthropic_health,
    is_anthropic_messages,
)
from agentix.bridge.plugins.anthropic.tokens import count_tokens
from agentix.bridge.plugins.openai.upstream import call_openai_compatible, openai_chat_completions_url
from agentix.bridge.types import ProxyAction
from agentix.bridge.util import trace


class BlockHostsInterceptor:
    priority = 10

    def intercept(self, ctx: RequestContext) -> ProxyAction | None:
        if blocked_host(ctx.host):
            trace("http.block", host=ctx.host, path=ctx.path)
            return {"action": "kill"}
        return None


class AnthropicHealthInterceptor:
    priority = 20

    def intercept(self, ctx: RequestContext) -> ProxyAction | None:
        if not is_anthropic_health(ctx):
            return None
        return respond_action(response_envelope(200, b""))


class AnthropicCountTokensInterceptor:
    priority = 30

    def intercept(self, ctx: RequestContext) -> ProxyAction | None:
        if not is_anthropic_count_tokens(ctx):
            return None
        return respond_action(json_envelope(count_tokens(ctx.parsed_body())))


class HookForwardInterceptor:
    priority = 100

    def intercept(self, ctx: RequestContext) -> ProxyAction | None:
        if not is_anthropic_messages(ctx):
            return None
        url = hook_url()
        if not url:
            return None
        body = ctx.parsed_body()
        event = http_request_event(ctx.flow, body)
        try:
            return send_hook_event(event, hook_url=url)
        except Exception as exc:
            trace("proxy_event.error", kind="http_request", error=repr(exc))
            return respond_action(response_envelope(502, f"abridge hook failed: {exc}".encode()))


class LocalUpstreamInterceptor:
    priority = 200

    def intercept(self, ctx: RequestContext) -> ProxyAction | None:
        if not is_anthropic_messages(ctx):
            return None
        if hook_url():
            return None
        body = ctx.parsed_body()
        client_stream = bool(body.get("stream"))
        openai_req = anthropic_to_openai_body(body)
        openai_req["stream"] = False
        trace(
            "anthropic.to_openai.host_request",
            target=openai_chat_completions_url(),
            model=openai_req["model"],
            client_stream=client_stream,
        )
        try:
            upstream = call_openai_compatible(openai_req)
        except Exception as exc:
            trace("anthropic.to_openai.host_error", error=repr(exc))
            return respond_action(response_envelope(502, f"openai-compatible upstream failed: {exc}".encode()))

        envelope = anthropic_client_envelope(body, upstream)
        trace(
            "anthropic.to_openai.host_response",
            client_stream=client_stream,
            bytes=len(envelope.get("body_base64", "")),
        )
        return respond_action(envelope)
