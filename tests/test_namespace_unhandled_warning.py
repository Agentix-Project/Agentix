"""An inbound event with no handler is the classic plugin footgun — a typo'd
`on_<event>` method or a mismatched namespace string drops it silently. The
namespace now warns once per event instead.
"""

from __future__ import annotations

import asyncio
import logging

from agentix.sio import Namespace


def test_dispatch_warns_once_for_unhandled_event(caplog) -> None:
    class NS(Namespace):
        namespace = "/warn-test"

        async def on_configure(self, data) -> None:  # the intended handler
            ...

    ns = NS()
    with caplog.at_level(logging.WARNING, logger="agentix.sio"):
        ns._dispatch("confgure", {})  # typo: should be 'configure'
        ns._dispatch("confgure", {})  # repeat — must not warn again

    hits = [r for r in caplog.records if "no handler" in r.getMessage()]
    assert len(hits) == 1
    msg = hits[0].getMessage()
    assert "/warn-test" in msg and "confgure" in msg
    # the message lists what IS registered, to point at the typo
    assert "configure" in msg


async def test_dispatch_does_not_warn_when_handler_exists(caplog) -> None:
    seen: list = []

    class NS(Namespace):
        namespace = "/ok-test"

        async def on_ping(self, data) -> None:
            seen.append(data)

    ns = NS()
    with caplog.at_level(logging.WARNING, logger="agentix.sio"):
        ns._dispatch("ping", {"x": 1})
        await asyncio.sleep(0)  # let the handler task run to completion

    assert seen == [{"x": 1}]
    assert not [r for r in caplog.records if "no handler" in r.getMessage()]
