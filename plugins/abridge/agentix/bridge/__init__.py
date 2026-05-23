"""agentix.bridge — extensible mitmproxy-backed sandbox traffic bridge."""

from __future__ import annotations

from typing import Any

from agentix.bridge.host.forwarder import HookForwarder
from agentix.bridge.host.pipeline import HostPipeline
from agentix.bridge.pipeline.registry import load_host_handlers, load_host_pipeline, load_interceptors
from agentix.bridge.plugins.anthropic.handler import AnthropicOpenAIHandler, OpenAIForwarder
from agentix.bridge.runtime.lifecycle import start_proxy, stop_proxy
from agentix.bridge.types import NAMESPACE, ProxyHandle

__version__ = "0.3.0"

__all__ = [
    "AnthropicOpenAIHandler",
    "HookForwarder",
    "HostPipeline",
    "NAMESPACE",
    "OpenAIForwarder",
    "ProxyHandle",
    "__version__",
    "load_host_handlers",
    "load_host_pipeline",
    "load_interceptors",
    "openai_forwarder",
    "start_proxy",
    "stop_proxy",
]


def openai_forwarder(
    *,
    base_url: str,
    api_key: str,
    model: str,
    extra_body: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> OpenAIForwarder:
    return OpenAIForwarder(
        base_url=base_url,
        api_key=api_key,
        model=model,
        extra_body=extra_body,
        timeout=timeout,
    )
