"""Default interceptor chain."""

from __future__ import annotations

from agentix.bridge.pipeline.interceptors import (
    AnthropicCountTokensInterceptor,
    AnthropicHealthInterceptor,
    BlockHostsInterceptor,
    HookForwardInterceptor,
    LocalUpstreamInterceptor,
)

DEFAULT_INTERCEPTORS = [
    BlockHostsInterceptor(),
    AnthropicHealthInterceptor(),
    AnthropicCountTokensInterceptor(),
    HookForwardInterceptor(),
    LocalUpstreamInterceptor(),
]
