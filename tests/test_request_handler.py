"""Tests for the host-side `request_handler` round-trip helper."""

from __future__ import annotations

from typing import Any

import pytest

from agentix import AsyncClientNamespace, request_handler

pytestmark = pytest.mark.asyncio


class _Host(AsyncClientNamespace):
    def __init__(self) -> None:
        super().__init__("/my-plugin")
        self.emitted: list[tuple[str, Any]] = []

    async def emit(self, event: str, data: Any = None, **kwargs: Any) -> Any:
        self.emitted.append((event, data))

    @request_handler("fetch")
    async def on_fetch(self, body: Any) -> Any:
        return {"echo": body}

    @request_handler("boom")
    async def on_boom(self, body: Any) -> Any:
        raise ValueError("nope")


async def test_request_handler_replies_with_result() -> None:
    host = _Host()
    await host.on_fetch({"request_id": "r1", "data": {"q": 1}})
    assert host.emitted == [("fetch:result", {"request_id": "r1", "value": {"echo": {"q": 1}}})]


async def test_request_handler_replies_with_error() -> None:
    host = _Host()
    await host.on_boom({"request_id": "r2", "data": 1})
    assert host.emitted == [
        ("boom:error", {"request_id": "r2", "error": {"type": "ValueError", "message": "nope"}}),
    ]
