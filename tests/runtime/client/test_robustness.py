"""Failure modes surface cheaply as typed errors — never a silent hang.

Three minimal-cost guarantees:
- `CallTimeout`        — a call exceeding the client deadline is bounded.
- `RuntimeUnreachable` — the server cannot be reached at connect time.
- `WorkerDied`         — the worker subprocess dies mid-call (crash / OOM).
"""

from __future__ import annotations

import socket

import pytest

from agentix.runtime.client.client import (
    CallTimeout,
    RemoteCallError,
    RuntimeClient,
    RuntimeUnreachable,
    WorkerExited,
)
from tests._worker_target import count_exec_and_sleep, self_sigkill


@pytest.mark.asyncio
async def test_call_timeout_when_call_exceeds_deadline(live_server):
    base_url = await live_server()
    # Warm up the worker with a deadline-less client so the multi-second
    # subprocess-spawn cost isn't billed against the tested deadline.
    async with RuntimeClient(base_url) as warm:
        await warm.remote(count_exec_and_sleep, 0.0)
    async with RuntimeClient(base_url, call_deadline=0.3) as c:
        with pytest.raises(CallTimeout):
            await c.remote(count_exec_and_sleep, 5.0)


@pytest.mark.asyncio
async def test_runtime_unreachable_when_server_down():
    # Grab a free port and immediately release it — nothing listens there.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    c = RuntimeClient(f"http://127.0.0.1:{port}")
    with pytest.raises(RuntimeUnreachable):
        await c.remote(count_exec_and_sleep, 0.0)
    await c.close()


@pytest.mark.asyncio
async def test_worker_death_surfaces_as_typed_error(live_server):
    base_url = await live_server()
    async with RuntimeClient(base_url, call_deadline=10) as c:
        await c.remote(count_exec_and_sleep, 0.0)  # warm up the worker
        with pytest.raises(RemoteCallError) as excinfo:
            await c.remote(self_sigkill)
    # The host gets a clean typed error, not a hang or a generic 500.
    assert excinfo.value.error.type == "WorkerDied"
    assert "signal 9" in excinfo.value.error.message
    # Surfaced as the public WorkerExited (a RemoteCallError subclass) with the
    # structured process exit status, so callers branch on OOM without string-matching.
    assert isinstance(excinfo.value, WorkerExited)
    assert excinfo.value.returncode == -9


@pytest.mark.asyncio
async def test_fail_pending_drains_queues_with_fatal_error():
    """On a terminal disconnect the client hands every in-flight call a fatal
    error so `remote(...)` stops waiting instead of hanging."""
    import asyncio

    client = RuntimeClient("http://127.0.0.1:1")
    try:
        q: asyncio.Queue = asyncio.Queue()
        client._pending["c1"] = q
        err = RuntimeUnreachable("connection lost")
        client._fail_pending(err)
        kind, data = q.get_nowait()
        assert kind == "fatal"
        assert data is err
    finally:
        await client._client.aclose()
