"""`Namespace._dispatch` must keep a strong reference to the task it spawns
for an async handler. asyncio only weakly references running tasks, so without
the `_detached_tasks` set a reply / ack / resume handler could be GC'd
mid-await — silently dropping the event and hanging the peer.
"""

from __future__ import annotations

import asyncio

import pytest

from agentix.sio import Namespace

pytestmark = pytest.mark.asyncio


async def test_dispatch_retains_and_releases_coroutine_handler_task() -> None:
    ran = asyncio.Event()

    class NS(Namespace):
        namespace = "/dispatch-test"

        async def on_ping(self, data) -> None:
            await asyncio.sleep(0)
            ran.set()

    ns = NS()
    ns._dispatch("ping", {})
    # Tracked while in flight (strong ref → cannot be GC'd mid-await).
    assert len(ns._detached_tasks) == 1
    await asyncio.wait_for(ran.wait(), timeout=1.0)
    # The done-callback discards it once complete (no unbounded growth).
    await asyncio.sleep(0)
    assert ns._detached_tasks == set()
