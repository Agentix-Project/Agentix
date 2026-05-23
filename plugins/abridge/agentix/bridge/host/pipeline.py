"""Host-side handler pipeline."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from agentix.bridge.actions import continue_action
from agentix.bridge.pipeline.protocols import HostHandler
from agentix.bridge.types import NAMESPACE, ProxyAction, ProxyEvent

from agentix import AsyncClientNamespace

logger = logging.getLogger("agentix.bridge.host")


class HostPipeline(AsyncClientNamespace):
    """Run an ordered chain of host handlers for proxy events."""

    def __init__(self, handlers: list[HostHandler], *, default: ProxyAction | None = None) -> None:
        super().__init__(NAMESPACE)
        self._handlers = handlers
        self._default = default or continue_action()

    async def on_proxy_event(self, payload: dict[str, Any]) -> None:
        req_id = payload.get("request_id")
        event = payload.get("data") or {}
        if not isinstance(req_id, str):
            logger.warning("abridge mitm: dropped proxy_event with no request_id")
            return
        if not isinstance(event, dict):
            event = {"kind": "bad_event", "raw": event}
        try:
            value = await self.handle_event(event)
        except Exception as exc:
            logger.exception("abridge mitm: proxy_event failed")
            await self.emit(
                "proxy_event:error",
                {
                    "request_id": req_id,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
            return
        await self.emit("proxy_event:result", {"request_id": req_id, "value": value or self._default})

    async def handle_event(self, event: ProxyEvent) -> ProxyAction | None:
        for handler in self._handlers:
            result = handler.handle(event)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return self._default
