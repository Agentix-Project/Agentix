"""Namespace round-trip: plugin sandbox-side namespace talks to a
plugin host-side namespace handler."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from agentix import AsyncClientNamespace, RuntimeClient
from tests._namespace_target import (
    echo_via_namespace,
    emit_formatted_log,
    emit_log_burst,
    emit_log_line,
    emit_log_with_exception,
    fire_namespace_event,
)
from tests._worker_target import print_stdout


class _EchoHost(AsyncClientNamespace):
    def __init__(self) -> None:
        super().__init__("/plugin-test")
        self.seen: list = []

    async def on_echo(self, data):
        self.seen.append(data)
        await self.emit(
            "echo:result",
            {
                "request_id": data["request_id"],
                "value": {"echoed": data["data"]},
            },
        )


@pytest.mark.asyncio
async def test_plugin_namespace_round_trip(live_server):
    base_url = await live_server()
    host_ns = _EchoHost()

    client = RuntimeClient(base_url)
    client.register_namespace(host_ns)
    async with client as c:
        result = await c.remote(echo_via_namespace, {"hello": 1})

    assert result == {"echoed": {"hello": 1}}
    assert len(host_ns.seen) == 1
    assert host_ns.seen[0]["data"] == {"hello": 1}


class _SlowHost(AsyncClientNamespace):
    """Host namespace whose `slow` handler blocks for a long time."""

    def __init__(self, hold: float) -> None:
        super().__init__("/plugin-test")
        self._hold = hold
        self.started = False
        self.finished = False

    async def on_slow(self, data):
        self.started = True
        await asyncio.sleep(self._hold)
        self.finished = True


@pytest.mark.asyncio
async def test_slow_namespace_handler_does_not_block_runtime(live_server):
    """A slow plugin handler must not stall the SIO receive loop —
    otherwise unrelated `c.remote` results queue up behind it.

    Regression: `socketio.AsyncClient` awaits `trigger_event` inline in
    its single websocket receive loop. `AsyncClientNamespace` detaches
    data-event handlers so a slow one can't freeze the connection.
    """
    base_url = await live_server()
    slow_host = _SlowHost(hold=30.0)

    client = RuntimeClient(base_url)
    client.register_namespace(slow_host)
    async with client as c:
        # Fire the event whose host handler sleeps 30s.
        await c.remote(fire_namespace_event, {"k": "v"})

        # Immediately do a normal RPC. If the slow handler blocked the
        # receive loop, this `call:result` would be stuck behind it for
        # ~30s. With the fix it returns near-instantly.
        t0 = time.perf_counter()
        result = await asyncio.wait_for(c.remote(abs, -5), timeout=10)
        elapsed = time.perf_counter() - t0

    assert result == 5
    assert elapsed < 8.0, f"runtime stalled behind slow handler: {elapsed:.1f}s"
    assert slow_host.started, "slow handler never ran"


# ── /log: raw stdout/stderr capture (Ray-style) ────────────────────────
#
# The worker captures its stdout and stderr (stdlib `logging` writes to
# stderr, so it is captured too) and streams each line best-effort on
# `/log`. The host replays each line under `agentix.sandbox.{stdout,stderr}`.


def _capture(logger_name: str) -> tuple[list[str], logging.Logger, logging.Handler]:
    captured: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    lg = logging.getLogger(logger_name)
    lg.setLevel(logging.INFO)
    handler = _Cap()
    lg.addHandler(handler)
    return captured, lg, handler


async def _await_line(captured: list[str], needle: str, *, timeout: float = 3.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if any(needle in m for m in captured):
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_user_logging_arrives_on_host_via_stderr(live_server):
    """Stdlib `logging` inside the sandbox writes to stderr, which the
    runtime captures and replays on the host under `agentix.sandbox.stderr`
    — including %-formatted messages and exception tracebacks."""
    base_url = await live_server()
    captured, lg, h = _capture("agentix.sandbox.stderr")
    try:
        async with RuntimeClient(base_url) as c:
            await c.remote(emit_log_line, "from sandbox", "INFO")
            await c.remote(emit_formatted_log, "user %s acted on %s", "alice", "doc-7")
            await c.remote(emit_log_with_exception, "caught one")
            assert await _await_line(captured, "from sandbox")
            assert await _await_line(captured, "user alice acted on doc-7")
            # logger.exception() writes the traceback to stderr too.
            assert await _await_line(captured, "ValueError: kaboom")
    finally:
        lg.removeHandler(h)


@pytest.mark.asyncio
async def test_remote_print_stdout_arrives_on_host(live_server):
    base_url = await live_server()
    captured, lg, h = _capture("agentix.sandbox.stdout")
    try:
        async with RuntimeClient(base_url) as c:
            result = await c.remote(print_stdout, "hello from print")
            assert result == "printed"
            assert await _await_line(captured, "hello from print")
    finally:
        lg.removeHandler(h)


@pytest.mark.asyncio
async def test_captured_log_stream_preserves_order(live_server):
    """Captured stderr lines arrive on the host in FIFO order — the pipe and
    drain are ordered. Best-effort: no acks, no replay."""
    base_url = await live_server()
    captured, lg, h = _capture("agentix.sandbox.stderr")
    burst_count = 50
    try:
        async with RuntimeClient(base_url) as c:
            await c.remote(emit_log_burst, "burst", burst_count)
            deadline = asyncio.get_event_loop().time() + 5
            while asyncio.get_event_loop().time() < deadline:
                if sum(1 for m in captured if "burst-" in m) >= burst_count:
                    break
                await asyncio.sleep(0.05)
    finally:
        lg.removeHandler(h)

    seq = [int(m.split("burst-")[1][:3]) for m in captured if "burst-" in m]
    assert seq == sorted(seq), f"out-of-order capture: {seq}"
    assert seq == list(range(burst_count)), f"lost lines: got {len(seq)} of {burst_count}"


@pytest.mark.asyncio
async def test_worker_log_context_can_be_configured_with_env(live_server, monkeypatch):
    """`AGENTIX_WORKER_LOG_CONTEXT` labels the worker's log lines; the label
    rides along in the captured text."""
    monkeypatch.setenv("AGENTIX_WORKER_LOG_CONTEXT", "custom-worker-{id}")
    base_url = await live_server()
    captured, lg, h = _capture("agentix.sandbox.stderr")
    try:
        async with RuntimeClient(base_url) as c:
            await c.remote(emit_log_line, "ctx-check", "INFO")
            assert await _await_line(captured, "ctx-check")
    finally:
        lg.removeHandler(h)

    line = next(m for m in captured if "ctx-check" in m)
    assert "custom-worker-" in line
