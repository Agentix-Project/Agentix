"""HTTP request interceptor dispatch."""

from __future__ import annotations

from typing import Any

from agentix.bridge.actions import apply_http_action
from agentix.bridge.pipeline.context import RequestContext
from agentix.bridge.pipeline.defaults import DEFAULT_INTERCEPTORS
from agentix.bridge.pipeline.protocols import Interceptor
from agentix.bridge.util import trace


class Dispatcher:
    def __init__(self, interceptors: list[Interceptor] | None = None) -> None:
        self._interceptors = sorted(
            interceptors if interceptors is not None else DEFAULT_INTERCEPTORS,
            key=lambda item: item.priority,
        )

    def dispatch(self, flow: Any) -> None:
        ctx = RequestContext.from_http_flow(flow)
        trace("http.request", method=ctx.method, host=ctx.host, path=ctx.path)
        for interceptor in self._interceptors:
            action = interceptor.intercept(ctx)
            if action is None:
                continue
            if apply_http_action(flow, action):
                return


_default: Dispatcher | None = None


def get_dispatcher() -> Dispatcher:
    global _default
    if _default is None:
        _default = Dispatcher()
    return _default


def dispatch_http_request(flow: Any) -> None:
    get_dispatcher().dispatch(flow)
