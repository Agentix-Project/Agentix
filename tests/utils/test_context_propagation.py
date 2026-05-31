"""The unified ambient context rides `c.remote(fn, ...)` across the real
sandbox-worker boundary: baggage set on the host is readable in the
worker, and the active trace scope is restored so a span opened inside
the remote `fn` nests under the host's current span.
"""

from __future__ import annotations

import pytest

from agentix import RuntimeClient, trace
from agentix.utils import context
from tests._context_target import observe_context


@pytest.mark.asyncio
async def test_context_and_trace_propagate_to_worker(live_server):
    base_url = await live_server()

    with trace.trace("host-workflow"):
        with trace.span("host.parent") as parent:
            with context.scope(tenant="acme", attempt=2):
                async with RuntimeClient(base_url) as c:
                    observed = await c.remote(observe_context)

    # Baggage crossed the boundary verbatim.
    assert observed["tenant"] == "acme"
    assert observed["attempt"] == 2

    # The worker saw the host's trace, and its span nested under the
    # host's current span — one unified trace tree across processes.
    assert observed["observed_trace_id"] == parent.trace_id
    assert observed["child_trace_id"] == parent.trace_id
    assert observed["child_parent_id"] == parent.span_id


@pytest.mark.asyncio
async def test_empty_context_is_unset_in_worker(live_server):
    """With no host context active, the worker sees clean defaults — the
    carrier is omitted from the wire entirely."""
    base_url = await live_server()

    context.clear()
    async with RuntimeClient(base_url) as c:
        observed = await c.remote(observe_context)

    assert observed["tenant"] is None
    assert observed["attempt"] is None
    # No host trace open → the worker's span gets its own synthetic trace,
    # not one inherited from the host.
    assert observed["child_parent_id"] is None
