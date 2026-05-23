"""Entry-point registry for bridge plugins."""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import Any

from agentix.bridge.host.pipeline import HostPipeline
from agentix.bridge.pipeline.defaults import DEFAULT_INTERCEPTORS
from agentix.bridge.pipeline.protocols import HostHandler, Interceptor
from agentix.bridge.plugins.anthropic.handler import AnthropicOpenAIHandler

logger = logging.getLogger("agentix.bridge.registry")

INTERCEPTOR_GROUP = "agentix.bridge.interceptor"
HANDLER_GROUP = "agentix.bridge.handler"

_BUILTIN_HANDLERS: dict[str, Callable[..., HostHandler]] = {
    "anthropic-openai": lambda **kwargs: AnthropicOpenAIHandler(**kwargs),
}


def load_interceptors(names: list[str] | None = None) -> list[Interceptor]:
    if names is None:
        return list(DEFAULT_INTERCEPTORS)
    out: list[Interceptor] = []
    eps = {ep.name: ep for ep in importlib.metadata.entry_points(group=INTERCEPTOR_GROUP)}
    for name in names:
        if name not in eps:
            raise KeyError(f"unknown bridge interceptor {name!r}")
        out.append(eps[name].load())
    return sorted(out, key=lambda item: item.priority)


def load_host_handlers(spec: str | list[str], **kwargs: Any) -> list[HostHandler]:
    names = [spec] if isinstance(spec, str) else list(spec)
    handlers: list[HostHandler] = []
    eps = {ep.name: ep for ep in importlib.metadata.entry_points(group=HANDLER_GROUP)}
    for name in names:
        if name in _BUILTIN_HANDLERS:
            handlers.append(_BUILTIN_HANDLERS[name](**kwargs))
            continue
        if name not in eps:
            raise KeyError(f"unknown bridge handler {name!r}")
        factory = eps[name].load()
        handlers.append(factory(**kwargs) if callable(factory) else factory)
    return handlers


def load_host_pipeline(spec: str | list[str] = "anthropic-openai", **kwargs: Any) -> HostPipeline:
    return HostPipeline(load_host_handlers(spec, **kwargs))
