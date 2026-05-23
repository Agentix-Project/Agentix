"""Multiple `c.remote(...)` calls multiplex through one worker.

Verifies (a) concurrency — gathered RPCs complete in ~duration rather
than n*duration; (b) trace propagation — each remote span arrives on
the host with the originating RPC's `call_id` stamped on its attrs.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agentix import RuntimeClient, trace
from tests._concurrent_target import emit_and_sleep


@pytest.mark.asyncio
async def test_concurrent_remote_calls_share_worker(live_server):
    base_url = await live_server()

    n = 5
    duration = 0.5

    collected_starts: list = []
    collected_ends: list = []

    class CaptureProcessor(trace.Processor):
        def on_span_start(self, s: trace.Span) -> None:
            if s.name == "concurrent_test":
                collected_starts.append(s)

        def on_span_end(self, s: trace.Span) -> None:
            if s.name == "concurrent_test":
                collected_ends.append(s)

    proc = CaptureProcessor()
    trace.add_processor(proc)

    try:
        async with RuntimeClient(base_url) as c:
            # Warm up: the first c.remote pays the worker subprocess
            # spawn + import cost (several seconds). Time gather() after
            # that so the wall-time measurement reflects RPC overhead +
            # actual concurrency, not one-time boot.
            await c.remote(emit_and_sleep, label="warmup", duration=0.0)
            # Wait for the warmup span to drain through the trace pipe
            # before clearing — otherwise it shows up as an extra event
            # in the gathered batch.
            await asyncio.sleep(0.3)
            collected_starts.clear()
            collected_ends.clear()

            wall_start = time.perf_counter()
            results = await asyncio.gather(
                *[c.remote(emit_and_sleep, label=f"r{i}", duration=duration) for i in range(n)]
            )
            wall_total = time.perf_counter() - wall_start

            # Let trace pipe drain (worker → server → SIO → host).
            await asyncio.sleep(0.5)
    finally:
        trace.remove_processor(proc)

    # Concurrency: dominated by one sleep, not n*sleep.
    assert wall_total < n * duration * 0.75, (
        f"calls appear serialized: wall={wall_total:.2f}s, expected <<{n * duration:.1f}s"
    )

    # Correctness: every result echoes its label.
    labels = sorted(r["label"] for r in results)
    assert labels == [f"r{i}" for i in range(n)]

    # Each call's span is observed on the host exactly once, with its
    # label round-tripping through attrs, and the call_id stamped on by
    # the worker's WireForwardProcessor.
    assert len(collected_ends) == n, f"expected {n} concurrent_test span_end events, got {len(collected_ends)}"

    labels_seen = {s.attrs["label"] for s in collected_ends}
    assert labels_seen == {f"r{i}" for i in range(n)}

    call_ids = {s.attrs.get("call_id") for s in collected_ends}
    assert None not in call_ids, "host received a remote span with no call_id stamped"
    assert len(call_ids) == n, f"expected {n} distinct call_ids, got {len(call_ids)}: {call_ids}"

    # All ended successfully.
    statuses = {s.status for s in collected_ends}
    assert statuses == {"ok"}, f"non-ok statuses seen: {statuses}"
