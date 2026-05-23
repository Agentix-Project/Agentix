"""Cross-process trace: worker emits a nested span tree, host receives
every span lifecycle via the SIO bridge, with the worker's RPC call_id
stamped onto each span's attrs.

This is the end-to-end smoke for the tracing system across the wire.
"""

from __future__ import annotations

import asyncio

import pytest

from agentix import RuntimeClient, trace
from tests._trace_target import make_subtree


@pytest.mark.asyncio
async def test_worker_span_tree_arrives_on_host(live_server):
    base_url = await live_server()

    collected_starts: list[trace.Span] = []
    collected_ends: list[trace.Span] = []

    class Capture(trace.Processor):
        def on_span_start(self, s: trace.Span) -> None:
            if s.name.startswith("worker."):
                collected_starts.append(s)

        def on_span_end(self, s: trace.Span) -> None:
            if s.name.startswith("worker."):
                collected_ends.append(s)

    cap = Capture()
    trace.add_processor(cap)

    try:
        async with RuntimeClient(base_url) as c:
            result = await c.remote(make_subtree, label="hello")
            # Let trace pipe drain.
            await asyncio.sleep(0.4)
    finally:
        trace.remove_processor(cap)

    assert result == {"ok": True, "label": "hello"}

    # We expect two span_start + two span_end on the host
    # (worker.outer + worker.inner).
    starts = {s.name: s for s in collected_starts}
    ends = {s.name: s for s in collected_ends}
    assert set(starts) == {"worker.outer", "worker.inner"}, f"missing span_start; got {sorted(starts)}"
    assert set(ends) == {"worker.outer", "worker.inner"}, f"missing span_end; got {sorted(ends)}"

    outer_end = ends["worker.outer"]
    inner_end = ends["worker.inner"]

    # All four lifecycle events share a single trace_id (one RPC = one
    # synthetic trace on worker side, since no `with trace.trace(...)`
    # was opened by user code).
    trace_ids = {s.trace_id for s in collected_ends}
    assert len(trace_ids) == 1, f"expected 1 trace_id, got {trace_ids}"

    # Parent linkage: inner.parent_id == outer.span_id.
    assert inner_end.parent_id == outer_end.span_id, (
        f"inner.parent_id={inner_end.parent_id} expected outer.span_id={outer_end.span_id}"
    )
    assert outer_end.parent_id is None

    # Attributes round-tripped (including the call_id stamped by the
    # worker's WireForwardProcessor) and status + events present.
    assert outer_end.attrs.get("label") == "hello"
    assert outer_end.attrs.get("call_id"), "call_id not stamped on outer span"
    assert inner_end.attrs.get("call_id") == outer_end.attrs["call_id"], (
        "inner and outer should share call_id (same RPC)"
    )
    assert inner_end.attrs.get("size") == 42
    assert inner_end.status == "ok"
    assert outer_end.status == "ok"
    assert [ev.name for ev in inner_end.events] == ["midpoint"]
    assert inner_end.events[0].attributes.get("note") == "halfway"
