"""Drive a demo batch rollout and stream its events.

This is the bridge between `agentix.runner` and the browser. It reuses the
TUI's demo provider and phase-tracing adapters, runs `run_rollouts`, and
forwards each lifecycle event through an async `send` callback as a plain dict
(the WebSocket handler serializes it as JSON; tests pass a list-appender). The
`send` shape:

    {"type": "start",  "n": N, "instances": [iid, ...]}
    {"type": "phase",  "iid": str, "phase": "setup"|"agent"|"score"}
    {"type": "result", **Rollout.to_dict()}        # one per instance
    {"type": "done",   "total": N, "resolved": R, "failed": F}
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from agentix.runner import Rollout, run_rollouts
from eval_tui._adapters import TracingAgent, TracingDataset, instance_id
from eval_tui.demo import DemoAgent, DemoDataset, DemoProvider

Send = Callable[[dict[str, Any]], Awaitable[None]]


async def stream_demo_run(
    send: Send,
    *,
    n: int = 24,
    n_concurrent: int = 4,
    seed: int = 0,
    dur_scale: float = 0.05,
) -> None:
    """Run a no-Docker demo rollout of `n` instances, emitting events via
    `send` as they happen. Returns when the batch is done."""
    dataset = DemoDataset(n, seed=seed, dur_scale=dur_scale)
    instances = dataset.instances()
    await send({"type": "start", "n": n, "instances": [instance_id(i) for i in instances]})

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def on_phase(iid: str, phase: str) -> None:
        queue.put_nowait({"type": "phase", "iid": iid, "phase": phase})

    def on_result(rollout: Rollout) -> None:
        queue.put_nowait({"type": "result", **rollout.to_dict()})

    task = asyncio.create_task(
        run_rollouts(
            dataset=TracingDataset(dataset, on_phase),
            agent=TracingAgent(DemoAgent(), on_phase),
            provider=DemoProvider(),
            bundle="demo",
            instances=instances,
            n_concurrent=n_concurrent,
            on_result=on_result,
        )
    )

    # Forward events as they land; stop once the run is done and the queue is
    # drained. The short timeout keeps us responsive without busy-waiting. The
    # try/finally guarantees the run is torn down if `send` raises (the client
    # disconnected) — otherwise the background rollout would leak, holding
    # sandboxes/slots after the consumer is gone.
    try:
        while not task.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            await send(event)

        rollouts = await task  # surface any unexpected error
        resolved = sum(1 for r in rollouts if r.resolved)
        await send(
            {
                "type": "done",
                "total": len(rollouts),
                "resolved": resolved,
                "failed": len(rollouts) - resolved,
            }
        )
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
