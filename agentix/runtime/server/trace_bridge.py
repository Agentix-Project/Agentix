"""Wires `agentix.trace.emit(...)` to the Socket.IO `trace` room.

Kept separate from `app.py` so the FastAPI / ASGI composition isn't
intermixed with the closure-side tracing hook. `install(sio)` is called
once at server module load, after the AsyncServer is constructed.
"""

from __future__ import annotations

import asyncio

import socketio

import agentix.trace as trace
from agentix.idents import CallId, PackageName
from agentix.runtime.events import TRACE, TRACES_ROOM
from agentix.runtime.models import TraceEvent


def install(sio: socketio.AsyncServer) -> None:
    """Register the trace emitter so closure impls' `trace.emit(...)` calls
    flow as Socket.IO `trace` events to subscribers in the TRACES_ROOM.

    Emission is best-effort and fire-and-forget — no awaiting from the
    sync logging-style emit() call path.
    """

    def _emit(kind: str, payload: dict, call_id: CallId | None, source: PackageName | None) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop on this thread → drop
        event = TraceEvent(
            kind=kind, payload=payload, timestamp=trace.now(),
            call_id=call_id, source=source,
        )
        loop.create_task(sio.emit(TRACE, event.model_dump(mode="json"), room=TRACES_ROOM))

    trace.register_sink(_emit)
