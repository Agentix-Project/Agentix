"""End-to-end tests for the subprocess worker path.

In-process tests (test_namespace_protocol.py) exercise the multiplexer
through its InProcessWorker backend — same protocol, no subprocess.
These tests use the real SubprocessWorker so the stdio framing, RPC
correlation, and trace-frame forwarding all run for real.

The target class lives in `tests/_worker_target.py` — a real importable
module so the worker subprocess can `import _worker_target` after we
add `tests/` to its PYTHONPATH.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from agentix.runtime.models import RemoteRequest
from agentix.runtime.multiplexer import NamespaceMultiplexer, _SubprocessWorker


TESTS_DIR = Path(__file__).parent


@pytest.fixture
def worker_env(monkeypatch):
    """Inject `tests/` into PYTHONPATH so the worker subprocess can find
    `_worker_target`. The framework's own modules are already importable
    from the runtime's site-packages.
    """
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(TESTS_DIR), existing] if existing else [str(TESTS_DIR)]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parts))


async def test_subprocess_worker_unary_round_trip(worker_env):
    """A real worker subprocess runs a method and returns the value."""
    multiplexer = NamespaceMultiplexer()
    # Manually register a subprocess entry (bypassing entry-point discovery).
    from agentix.runtime.multiplexer import _NamespaceEntry
    multiplexer._entries["_worker_target"] = _NamespaceEntry(
        package="_worker_target", dist_name="test-worker", dist_version="0.0.0",
        target="_worker_target:Echo", python=sys.executable,
    )
    try:
        resp = await multiplexer.dispatch_unary(RemoteRequest(
            package="_worker_target", method="echo", kwargs={"msg": "hi"},
        ))
        assert resp.ok, resp.error
        assert resp.value == {"msg": "echo:hi"}
    finally:
        await multiplexer.shutdown()


async def test_subprocess_worker_streaming(worker_env):
    """Server-streaming method round-trip via subprocess."""
    multiplexer = NamespaceMultiplexer()
    from agentix.runtime.multiplexer import _NamespaceEntry
    multiplexer._entries["_worker_target"] = _NamespaceEntry(
        package="_worker_target", dist_name="test-worker", dist_version="0.0.0",
        target="_worker_target:Echo", python=sys.executable,
    )
    try:
        events = []
        async for ev in multiplexer.dispatch_stream(RemoteRequest(
            package="_worker_target", method="counter", kwargs={"n": 3},
        )):
            events.append(ev)
            if ev.get("type") in ("end", "error"):
                break
        items = [e["value"] for e in events if e.get("type") == "item"]
        assert items == [0, 1, 2]
        assert events[-1] == {"type": "end"}
    finally:
        await multiplexer.shutdown()


async def test_subprocess_worker_trace_forwarding(worker_env):
    """trace.emit() in the worker reaches the runtime's trace_forwarder."""
    received: list[tuple[str, dict]] = []

    def forwarder(kind, payload, call_id, source):
        received.append((kind, payload))

    multiplexer = NamespaceMultiplexer(trace_forwarder=forwarder)
    from agentix.runtime.multiplexer import _NamespaceEntry
    multiplexer._entries["_worker_target"] = _NamespaceEntry(
        package="_worker_target", dist_name="test-worker", dist_version="0.0.0",
        target="_worker_target:Echo", python=sys.executable,
    )
    try:
        resp = await multiplexer.dispatch_unary(RemoteRequest(
            package="_worker_target", method="trace_then_echo", kwargs={"msg": "x"},
        ))
        assert resp.ok, resp.error
        # Give the worker a moment to flush the trace frame, which is
        # fire-and-forget on the worker side; the read loop picks it up
        # but it may land just after the result.
        import asyncio
        await asyncio.sleep(0.2)
        assert ("test_event", {"msg": "x"}) in received
    finally:
        await multiplexer.shutdown()
