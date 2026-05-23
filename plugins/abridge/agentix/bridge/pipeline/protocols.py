"""Extension interfaces for the bridge pipeline."""

from __future__ import annotations

from typing import Protocol

from agentix.bridge.pipeline.context import RequestContext
from agentix.bridge.types import ProxyAction, ProxyEvent


class Interceptor(Protocol):
    priority: int

    def intercept(self, ctx: RequestContext) -> ProxyAction | None:
        """Return an action to stop the chain, or None to continue."""


class HostHandler(Protocol):
    async def handle(self, event: ProxyEvent) -> ProxyAction | None:
        """Return an action, or None to delegate to the next handler."""
