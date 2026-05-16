"""Wires `agentix.trace.emit(...)` to the Socket.IO `trace` room.

Trace events reach the runtime two ways:

  1. **In-process workers** (test fixtures) call `agentix.trace.emit` in
     the runtime's own process — the local `_sinks` list fires, including
     the SIO emitter we register below.
  2. **Subprocess workers** (production) call `agentix.trace.emit` in
     a child process, which forwards `trace` frames over stdio. The
     multiplexer's `_trace_forwarder` callback (the same function we
     return from `install_trace_bridge`) is called for each frame and
     re-emits onto the SIO `traces` room.

Both paths converge in one function — the SIO room is the single
publication channel.
"""

from __future__ import annotations

import asyncio
from typing import Any

import socketio

import agentix.trace as trace
from agentix.idents import CallId, PackageName
from agentix.runtime.events import TRACE, TRACES_ROOM
from agentix.runtime.models import TraceEvent


def install_trace_bridge(sio: socketio.AsyncServer):
    """Register the SIO-emitting sink and return the callback for the
    multiplexer's `trace_forwarder` to call.

    Emission is best-effort and fire-and-forget — no awaiting from the
    sync logging-style emit() call path.
    """

    def _emit(
        kind: str, payload: dict[str, Any],
        call_id: CallId | str | None, source: PackageName | str | None,
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop on this thread → drop
        event = TraceEvent(
            kind=kind, payload=payload, timestamp=trace.now(),
            call_id=call_id, source=source,  # type: ignore[arg-type]
        )
        loop.create_task(sio.emit(TRACE, event.model_dump(mode="json"), room=TRACES_ROOM))

    # Path 1: in-process workers' trace.emit() flows through this sink.
    trace.register_sink(_emit)
    # Path 2: returned to the multiplexer so subprocess workers' trace
    # frames go through the same emitter.
    return _emit
