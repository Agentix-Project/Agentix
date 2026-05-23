"""End-to-end tests for the runtime's "reconnect and lose nothing"
contract.

Both the RPC channel (`c.remote(...)`) and the side-channel streams
(`/log`, `/trace`) are designed to recover transparently when the SIO
transport drops, as long as the underlying server process stays alive.
These tests simulate an involuntary disconnect by force-closing the
EngineIO transport (without going through the voluntary-disconnect
codepath that socketio uses for `client.disconnect()`), then assert
that:

  - In-flight `c.remote(...)` calls still return their result via the
    `resume` / `ack` protocol once socketio's auto-reconnect succeeds.
  - Records emitted on `/log` while no host was connected still arrive
    after reconnect, in original order, exactly once.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from agentix import RuntimeClient
from agentix.log._config import LOG_CONTEXT_ATTR
from tests import _worker_target as target
from tests._namespace_target import emit_log_burst

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


async def _wait_until(predicate, *, timeout: float = 5.0, step: float = 0.05) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(step)
    return predicate()


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


async def test_log_stream_resumes_records_buffered_during_disconnect(
    use_inprocess_worker, live_server
):
    """`/log` records the worker emits while the host is offline must
    arrive after reconnect, with no duplicates and in FIFO order."""
    use_inprocess_worker()
    base_url = await live_server()

    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.name == "namespace_target":
                captured.append(record)

    target_logger = logging.getLogger("namespace_target")
    target_logger.setLevel(logging.INFO)
    handler = _Capture()
    target_logger.addHandler(handler)

    burst_count = 30
    try:
        async with RuntimeClient(base_url) as c:
            # Kick off the burst; it returns once the worker has
            # finished emitting all records (queued on the worker's
            # outbound pipe). The host SIO loop drains them
            # asynchronously, so dropping the connection right after
            # remote() returns leaves at least some records still in
            # the sandbox-side `ReliableStream` buffer.
            burst = asyncio.create_task(c.remote(emit_log_burst, "burst", burst_count))

            # Give the worker a moment to start producing records.
            await asyncio.sleep(0.05)
            await _force_disconnect(c._sio)
            # Make sure the burst remote() call itself finishes (its
            # result rides the same resume protocol).
            assert await asyncio.wait_for(burst, timeout=15) == burst_count

            # Wait for every record to land on the host.
            ok = await _wait_until(
                lambda: sum(1 for r in captured if r.getMessage().startswith("burst-"))
                >= burst_count,
                timeout=15,
            )
            assert ok, (
                f"only {sum(1 for r in captured if r.getMessage().startswith('burst-'))}"
                f" of {burst_count} records arrived"
            )
    finally:
        target_logger.removeHandler(handler)

    messages = [r.getMessage() for r in captured if r.getMessage().startswith("burst-")]
    expected = [f"burst-{i:03d}" for i in range(burst_count)]
    assert messages == expected, f"out-of-order or duplicate delivery: {messages[:5]}..."

    # Sanity: every record carries the same sandbox-side log context,
    # confirming they all came over the same `/log` stream rather than
    # bypassing the bridge.
    contexts = {getattr(r, LOG_CONTEXT_ATTR, "") for r in captured if r.getMessage().startswith("burst-")}
    assert len(contexts) == 1
    assert next(iter(contexts)).startswith("sandbox-")
