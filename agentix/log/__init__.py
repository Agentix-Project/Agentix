"""agentix.log — sandbox-side `logging` records ferried to the host.

This module is a thin bridge for the *third* observability pillar
(distinct from `agentix.trace`). Workers don't need a custom API: just
use stdlib `logging`:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("hello from sandbox")

At worker boot, `install_worker_bridge()` adds a `logging.Handler` to
the root logger that emits each `LogRecord` on the `/log` SIO
namespace. The host's `RuntimeClient` auto-registers a consumer that
forwards records into the host's own `logging` system, so they appear
in host logs untouched.

## Delivery contract

`/log` is a side channel, separate from the `c.remote(...)` result
path. The contract is:

  - **Ordering**: records emitted on a single connection arrive in
    FIFO order.
  - **Eventual delivery**: under a healthy connection, every emitted
    record reaches the host.
  - **No happens-before with `remote()`**: a log record emitted from
    inside `fn` may arrive on the host *after* `c.remote(fn, ...)`
    has already returned. Treat side-channel observability as
    eventually-consistent telemetry, not as a synchronization barrier.
"""

from __future__ import annotations

import logging

from agentix.log._config import configure_logging

__all__ = ["configure_logging", "install_worker_bridge"]


def install_worker_bridge(level: int = logging.NOTSET) -> logging.Handler:
    """Install the bridge handler on the root logger. Idempotent."""
    from agentix.log._bridge import WorkerLogHandler

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, WorkerLogHandler):
            return h
    handler = WorkerLogHandler()
    handler.setLevel(level)
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return handler
