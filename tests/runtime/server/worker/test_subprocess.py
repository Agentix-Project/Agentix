"""End-to-end tests for the subprocess worker path.

Protocol tests exercise the worker client without subprocess stdio.
These tests use the real subprocess worker so the stdio framing and
call correlation run for real.

The target module lives in `tests/_worker_target.py`.
"""

from __future__ import annotations

import asyncio
import pickle
import sys
from pathlib import Path

import pytest

from agentix.runtime.server.worker import RuntimeWorkerClient
from agentix.runtime.shared.models import RemoteRequest
from tests import _worker_target as target
from tests._rpc_helpers import request_for


def _make_worker() -> RuntimeWorkerClient:
    mp = RuntimeWorkerClient()
    mp._python = sys.executable
    return mp


async def test_subprocess_worker_round_trip():
    """A real worker subprocess runs a callable and returns the value."""
    mp = _make_worker()
    try:
        resp = await mp.call(request_for(target.echo, kwargs={"msg": "hi"}))
        assert resp.ok, resp.error
        result = pickle.loads(resp.value)
        assert result.msg == "echo:hi"
    finally:
        await mp.shutdown()


async def test_subprocess_worker_handles_many_concurrent_calls():
    """Smoke test: many concurrent calls funnel through the single outbound
    drainer without deadlock and every result comes back correctly.

    This exercises the queue design under concurrency; it is not a strict
    regression guard against the old lock-across-`drain()` (which only
    deadlocks under sustained two-way pipe backpressure, impractical to force
    deterministically with a real subprocess here)."""
    mp = _make_worker()
    try:
        responses = await asyncio.gather(
            *(mp.call(request_for(target.echo, kwargs={"msg": str(i)})) for i in range(25))
        )
        msgs = sorted(pickle.loads(r.value).msg for r in responses)
        assert msgs == sorted(f"echo:{i}" for i in range(25))
    finally:
        await mp.shutdown()


async def test_subprocess_worker_bad_callable_fails_fast():
    """A garbage `callable` string yields an error, not a hang."""
    from agentix.runtime.shared.callables import RemoteCallable

    mp = _make_worker()
    try:
        resp = await asyncio.wait_for(
            mp.call(
                RemoteRequest(
                    callable=RemoteCallable("not-valid-import-path"),
                    arguments=pickle.dumps(((), {})),
                )
            ),
            timeout=20,
        )
    finally:
        await mp.shutdown()
    assert not resp.ok
    assert resp.error is not None


async def test_subprocess_worker_ignores_cwd_agentix_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = tmp_path / "agentix"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise RuntimeError('cwd agentix package imported')\n")
    monkeypatch.chdir(tmp_path)

    mp = _make_worker()
    try:
        resp = await mp.call(request_for(target.echo, kwargs={"msg": "shadow"}))
        assert resp.ok, resp.error
        result = pickle.loads(resp.value)
        assert result.msg == "echo:shadow"
    finally:
        await mp.shutdown()


async def test_subprocess_worker_child_reading_stdin_does_not_steal_frames():
    """A remote call that spawns a child reading stdin (like the `claude`
    CLI) must not consume control-pipe bytes — later calls still work.

    Guards the fd-isolation invariant: the worker repoints fd 0 to
    /dev/null so inherited stdin is harmless to the frame stream.
    """
    mp = _make_worker()
    try:
        r1 = await mp.call(request_for(target.spawn_stdin_reading_child))
        assert r1.ok, r1.error
        assert pickle.loads(r1.value) == 0

        # If the child had stolen frame bytes, the pipe would be desynced
        # and this call would hang or return garbage.
        r2 = await mp.call(request_for(target.echo, kwargs={"msg": "after"}))
        assert r2.ok, r2.error
        assert pickle.loads(r2.value).msg == "echo:after"
    finally:
        await mp.shutdown()


async def test_subprocess_worker_death_fails_in_flight_call():
    """Killing the worker mid-call surfaces WorkerExited to the caller."""
    mp = _make_worker()
    try:
        # Warm the worker.
        resp = await mp.call(request_for(target.echo, kwargs={"msg": "warm"}))
        assert resp.ok

        # Start a long-running call, kill the worker mid-execution.
        import asyncio as _asyncio

        from agentix.runtime.shared.callables import RemoteCallable

        slow = RemoteRequest(
            callable=RemoteCallable._resolve(_asyncio.sleep),
            arguments=pickle.dumps(((30.0,), {})),
        )
        task = asyncio.create_task(mp.call(slow))
        await asyncio.sleep(0.2)

        worker = mp._worker
        assert worker is not None
        proc = worker._proc  # type: ignore[attr-defined]
        assert proc is not None
        proc.kill()

        with pytest.raises(RuntimeError, match="runtime worker exited"):
            await asyncio.wait_for(task, timeout=5)
    finally:
        await mp.shutdown()
