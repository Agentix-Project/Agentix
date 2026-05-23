"""Anthropic host/path matchers."""

from __future__ import annotations

from agentix.bridge.pipeline.context import RequestContext
from agentix.bridge.util import csv_env, host_matches


def anthropic_hosts() -> list[str]:
    return csv_env("ABRIDGE_ANTHROPIC_HOSTS", "api.anthropic.com,localhost,127.0.0.1")


def is_anthropic_host(host: str) -> bool:
    return _host_matches(host, anthropic_hosts())


def _host_matches(host: str, patterns: list[str]) -> bool:
    return host_matches(host, patterns)


def is_anthropic_health(ctx: RequestContext) -> bool:
    return is_anthropic_host(ctx.host) and ctx.path_without_query == "/"


def is_anthropic_count_tokens(ctx: RequestContext) -> bool:
    return (
        ctx.method == "POST"
        and is_anthropic_host(ctx.host)
        and ctx.path_without_query == "/v1/messages/count_tokens"
    )


def is_anthropic_messages(ctx: RequestContext) -> bool:
    return (
        ctx.method == "POST"
        and is_anthropic_host(ctx.host)
        and ctx.path_without_query == "/v1/messages"
    )


def blocked_host(host: str) -> bool:
    patterns = csv_env("ABRIDGE_BLOCK_HOSTS", "segment.io,telemetry")
    return any(pattern and pattern in host for pattern in patterns)
