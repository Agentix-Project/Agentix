"""End-to-end test for the runtime's "reconnect and lose nothing"
contract on the RPC channel.

An in-flight `c.remote(...)` must still return its result when the SIO
transport drops, as long as the server process stays alive: the server
keeps the task running across the disconnect, caches the terminal result
in `pending_results`, and the reconnecting client emits `resume` to pick
it up. We simulate an involuntary disconnect by force-closing the
EngineIO transport (not the voluntary-disconnect codepath socketio uses
for `client.disconnect()`).

(`/log` is a best-effort live stream — no resume/replay — so it is not
part of this contract; durable capture is the sandbox-side file.)
"""

from __future__ import annotations

import asyncio

import pytest

from agentix import RuntimeClient
from tests import _worker_target as target

pytestmark = pytest.mark.asyncio


async def _force_disconnect(sio) -> None:
    """Tear the underlying websocket transport from under the client
    so that socketio sees an unscheduled drop and triggers its
    auto-reconnect path. `eio.disconnect()` is a voluntary close —
    socketio explicitly skips reconnect on those — so we close the
    transport directly. This mirrors the real failure mode (a TCP
    blip / server restart on the same port)."""
    assert sio is not None and sio.connected
    ws = sio.eio.ws  # underlying aiohttp ClientWebSocketResponse
    await ws.close()


async def test_in_flight_remote_call_resumes_after_disconnect(use_inprocess_worker, live_server):
    """A `c.remote(...)` that's mid-flight when the transport drops
    must still return its result once the client auto-reconnects.

    Mechanism: server keeps the task running across the disconnect,
    caches the terminal `(event, frame)` in `pending_results`, and the
    reconnecting client emits `resume` so it picks up the cached
    result via SIO.
    """
    use_inprocess_worker()
    base_url = await live_server()

    target._exec_counter = 0

    async with RuntimeClient(base_url) as c:
        # 1.5s call: long enough that we can drop the link mid-flight
        # but short enough not to slow CI noticeably.
        remote_task = asyncio.create_task(c.remote(target.count_exec_and_sleep, 1.5))

        # Let the call land on the server before we yank the cable.
        await asyncio.sleep(0.3)
        await _force_disconnect(c._sio)

        # The remote task should still finish — even though the
        # underlying transport was dropped, both sides recover and the
        # cached result reaches the client over the resumed channel.
        result = await asyncio.wait_for(remote_task, timeout=15)

    assert result == 1, "fn must have run exactly once across the disconnect"
