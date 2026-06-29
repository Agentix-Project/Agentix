"""`/log` — best-effort raw stdout/stderr capture from the sandbox.

The worker captures its own stdout and stderr (Ray-style; stdlib
`logging` writes to stderr, so it is captured too) and streams each line
best-effort on the `/log` namespace as a `line` event carrying
`{stream, line}`. The host replays each line into its own `logging` tree
under `agentix.sandbox.{stdout,stderr}`, so it shows up in host logs.

This channel is the live, lossy stream — no acks, no replay. Durable
capture is the sandbox-side file the worker also writes.
"""

from __future__ import annotations

import logging
from typing import Any

import socketio

LOG_NAMESPACE = "/log"
LOG_EVENT = "line"


class HostLogNamespace(socketio.AsyncClientNamespace):
    """Replays inbound `/log:line` events into `agentix.sandbox.{stream}`.

    Replay runs INLINE in the receive loop (we override `trigger_event`
    directly rather than inheriting agentix's detached-dispatch
    `AsyncClientNamespace`). This is deliberate and mirrors the sibling
    `HostTraceNamespace`: the handler is a single non-blocking
    `logging.getLogger(...).info(line)`, and inline replay preserves strict
    FIFO line order for free. The only way it could stall the loop is a
    user-installed *slow* handler on the `agentix.sandbox.*` loggers — an
    unusual setup; route such handlers through a `QueueHandler` if needed."""

    def __init__(self) -> None:
        super().__init__(LOG_NAMESPACE)

    async def trigger_event(self, event: str, *args: Any) -> Any:
        if event != LOG_EVENT:
            return
        from agentix.runtime.client._sio_facade import _decode

        payload = _decode(args[0]) if args else None
        if not isinstance(payload, dict):
            return
        line = payload.get("line")
        if not isinstance(line, str):
            return
        stream = str(payload.get("stream", "stdout"))
        logging.getLogger(f"agentix.sandbox.{stream}").info(line)


__all__ = ["LOG_EVENT", "LOG_NAMESPACE", "HostLogNamespace"]
