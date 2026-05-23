"""Generic host-side hook receiver."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable

from agentix.bridge.actions import continue_action
from agentix.bridge.types import NAMESPACE, ProxyAction, ProxyEvent

from agentix import AsyncClientNamespace

logger = logging.getLogger("agentix.bridge.host")
HookHandler = Callable[[ProxyEvent], ProxyAction | Awaitable[ProxyAction | None] | None]


class HookForwarder(AsyncClientNamespace):
    """Host-side receiver for protocol-neutral mitmproxy hook events."""

    def __init__(self, handler: HookHandler | None = None) -> None:
        super().__init__(NAMESPACE)
        self._handler = handler

    async def on_proxy_event(self, payload: dict[str, object]) -> None:
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
        await self.emit("proxy_event:result", {"request_id": req_id, "value": value or continue_action()})

    async def handle_event(self, event: ProxyEvent) -> ProxyAction | None:
        if self._handler is None:
            return continue_action()
        result = self._handler(event)
        if inspect.isawaitable(result):
            return await result
        return result
