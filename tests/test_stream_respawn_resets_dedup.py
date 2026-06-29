"""Regression: a worker respawn must not silently drop the new stream.

The host bridges (`/trace`, `/log`) dedup by `_seq`: an inbound event with
`seq <= _last_seq` is treated as a duplicate of an already-delivered event
and dropped. When a worker subprocess crashes and is respawned, it builds a
fresh `ReliableStream` whose `_seq` counter restarts at 1 — but the host's
`_last_seq` is still high from the previous worker, so without a stream id
the new worker's first events look like duplicates and vanish.

`ReliableStream` stamps each envelope with a per-instance `_sid`; the host
resets its cursor when the id changes. These tests feed two streams (the
second restarting its `_seq`) straight through `trigger_event` and assert
the restarted stream's early events are delivered, not swallowed.
"""

from __future__ import annotations

from agentix.runtime.shared.codec import pack
from agentix.utils.trace import _bridge as trace_bridge


def _env(sid: str, seq: int, data: dict) -> bytes:
    # Matches the server's msgpack framing that the host bridges `_decode`.
    return pack({"_sid": sid, "_seq": seq, "data": data})


async def test_trace_respawn_resets_dedup_cursor(monkeypatch) -> None:
    dispatched: list[tuple[str, dict]] = []
    monkeypatch.setattr(trace_bridge, "_dispatch", lambda event, payload: dispatched.append((event, payload)))

    ns = trace_bridge.HostTraceNamespace()

    # Stream A climbs to seq 3.
    for seq in (1, 2, 3):
        await ns.trigger_event("span_start", _env("aaaa", seq, {"span_id": f"a{seq}"}))
    assert [p["span_id"] for _, p in dispatched] == ["a1", "a2", "a3"]
    assert ns._last_seq == 3

    # Worker respawns: a NEW stream id whose seq restarts at 1. Pre-fix this is
    # dropped as a duplicate (1 <= 3); it must be delivered.
    await ns.trigger_event("span_start", _env("bbbb", 1, {"span_id": "b1"}))
    assert dispatched[-1] == ("span_start", {"span_id": "b1"})
    assert ns._sid == "bbbb"
    assert ns._last_seq == 1


async def test_trace_same_stream_still_dedups(monkeypatch) -> None:
    # The reset must not weaken in-stream dedup: a replayed lower seq with the
    # SAME id is still a duplicate and must be dropped.
    dispatched: list[tuple[str, dict]] = []
    monkeypatch.setattr(trace_bridge, "_dispatch", lambda event, payload: dispatched.append((event, payload)))

    ns = trace_bridge.HostTraceNamespace()
    for seq in (1, 2, 3):
        await ns.trigger_event("span_start", _env("aaaa", seq, {"span_id": f"a{seq}"}))
    await ns.trigger_event("span_start", _env("aaaa", 2, {"span_id": "dup"}))  # resume replay

    assert [p["span_id"] for _, p in dispatched] == ["a1", "a2", "a3"]  # no "dup"
