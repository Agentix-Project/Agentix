"""RuntimeClient construction options."""

from __future__ import annotations

import socketio

from agentix import RuntimeClient


def test_http_sync_ms_default() -> None:
    client = RuntimeClient("http://localhost:0")
    assert client._http_sync_budget_ms == 1000


async def test_http_sync_ms_none_disables_fast_path() -> None:
    client = RuntimeClient("http://localhost:0", http_sync_ms=None)
    try:
        kind, value = await client._try_http_fast_path(sio=socketio.AsyncClient(), payload={})
        assert kind == "fallback"
        assert value is None
    finally:
        await client.close()
