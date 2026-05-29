"""Unit tests for `agentix.sio.ReliableStream`.

The stream guarantees three properties for a sandbox-side namespace:

  - Each emit gets a strictly monotonic `_seq`.
  - Buffered emits stay around until the host confirms via `_ack`.
  - On `_resume {since_seq}`, every buffered emit with seq > since_seq
    is replayed in original FIFO order.

These tests drive the protocol directly (no SIO server required) by
installing a fake send hook on the bridge and calling the inbound
dispatcher manually. All tests run under an event loop so that the
namespace's async dispatch path (`asyncio.create_task` for handler
coroutines) has somewhere to attach.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from agentix import sio as _sio

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _isolated_bridge():
    """Each test gets a clean module-level bridge so namespaces from
    one test don't leak into the next."""
    sent: list[dict[str, Any]] = []
    _sio._bridge._namespaces = {}
    _sio._bridge.install(sent.append)
    try:
        yield sent
    finally:
        _sio._bridge._send = None
        _sio._bridge._namespaces = {}


def _emitted(sent: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    """Filter out the `sio_open` registration frame; keep emits of `event`."""
    return [f for f in sent if f.get("type") == "sio_emit" and f.get("event") == event]


async def test_seq_monotonic_and_envelope_shape(_isolated_bridge):
    sent = _isolated_bridge
    ns = _sio.Namespace("/test")
    _sio.register_namespace(ns)
    stream = _sio.ReliableStream(ns)

    seqs = [stream.emit_nowait("record", {"i": i}) for i in range(5)]
    assert seqs == [1, 2, 3, 4, 5]

    frames = _emitted(sent, "record")
    assert len(frames) == 5
    for i, frame in enumerate(frames):
        assert frame["data"] == {"_seq": i + 1, "data": {"i": i}}


async def test_ack_releases_buffered_entries(_isolated_bridge):
    ns = _sio.Namespace("/test")
    _sio.register_namespace(ns)
    stream = _sio.ReliableStream(ns)

    for i in range(5):
        stream.emit_nowait("record", {"i": i})
    assert len(stream._buffer) == 5

    await stream._on_ack({"seq": 3})
    assert [seq for seq, _, _ in stream._buffer] == [4, 5]

    await stream._on_ack({"seq": 99})
    assert len(stream._buffer) == 0


async def test_resume_replays_only_unacked(_isolated_bridge):
    sent = _isolated_bridge
    ns = _sio.Namespace("/test")
    _sio.register_namespace(ns)
    stream = _sio.ReliableStream(ns)

    for i in range(4):
        stream.emit_nowait("record", {"i": i})
    await stream._on_ack({"seq": 2})

    sent.clear()
    await stream._on_resume({"since_seq": 0})

    replayed = _emitted(sent, "record")
    assert [f["data"]["_seq"] for f in replayed] == [3, 4]


async def test_resume_with_since_seq_filters_already_seen(_isolated_bridge):
    sent = _isolated_bridge
    ns = _sio.Namespace("/test")
    _sio.register_namespace(ns)
    stream = _sio.ReliableStream(ns)

    for i in range(5):
        stream.emit_nowait("record", {"i": i})

    sent.clear()
    await stream._on_resume({"since_seq": 3})

    replayed = _emitted(sent, "record")
    assert [f["data"]["_seq"] for f in replayed] == [4, 5]


async def test_wrap_runs_under_the_lock(_isolated_bridge):
    """`_wrap` must perform its seq allocation + buffer mutation under
    `_lock`. Hold the lock and a concurrent `emit_nowait` blocks inside
    `_wrap` until it's released. This is a deterministic regression guard:
    on the unsynchronized version `_wrap` doesn't take the lock, so the
    emit would complete immediately instead of blocking.

    (A plain "do concurrent emits collide?" stress test is NOT a useful
    guard here: on GIL CPython the short read-modify-write of `_next_seq`
    rarely interleaves, so it passes with or without the lock. The lock's
    value is correctness by construction and safety under free-threaded /
    no-GIL CPython, where the RMW and the deque op are not atomic.)"""
    ns = _sio.Namespace("/test")
    _sio.register_namespace(ns)
    stream = _sio.ReliableStream(ns)

    done = threading.Event()

    def emit_one() -> None:
        stream.emit_nowait("record", {})
        done.set()

    stream._lock.acquire()
    worker = threading.Thread(target=emit_one)
    try:
        worker.start()
        # While we hold the lock, the emit cannot finish — it is parked on
        # `_lock` inside `_wrap`.
        assert not done.wait(timeout=0.3), "emit completed without taking _lock"
    finally:
        stream._lock.release()

    worker.join(timeout=2)
    assert done.is_set()
    assert [seq for seq, _, _ in stream._buffer] == [1]
    assert stream._next_seq == 2


async def test_buffer_cap_drops_oldest(_isolated_bridge):
    ns = _sio.Namespace("/test")
    _sio.register_namespace(ns)
    stream = _sio.ReliableStream(ns, max_buffer=3)

    for i in range(5):
        stream.emit_nowait("record", {"i": i})

    # deque(maxlen=3): only the latest three survive.
    assert [seq for seq, _, _ in stream._buffer] == [3, 4, 5]


async def test_dispatch_routes_ack_and_resume_through_namespace(_isolated_bridge):
    """Inbound `_ack` / `_resume` events from the host hit the
    Namespace's dispatch path; the stream wires them via `on()`. This
    proves the end-to-end sandbox-side wiring works without needing
    the network layer."""
    sent = _isolated_bridge
    ns = _sio.Namespace("/test")
    _sio.register_namespace(ns)
    stream = _sio.ReliableStream(ns)

    for i in range(3):
        stream.emit_nowait("record", {"i": i})

    # Drive an inbound _ack frame as if it arrived from the host.
    _sio._dispatch_inbound("/test", "_ack", {"seq": 2})
    # Dispatch schedules the async handler; let the loop run it.
    await asyncio.sleep(0)

    assert [seq for seq, _, _ in stream._buffer] == [3]

    sent.clear()
    _sio._dispatch_inbound("/test", "_resume", {"since_seq": 2})
    await asyncio.sleep(0)

    replayed = _emitted(sent, "record")
    assert [f["data"]["_seq"] for f in replayed] == [3]
