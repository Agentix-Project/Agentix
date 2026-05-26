"""agentix.utils.log — sandbox-side logs ferried to the host.

This module is a thin bridge for the *third* observability pillar
(distinct from `agentix.utils.trace`). Workers don't need a custom API:
stdout is captured by the runtime, and stdlib logging is bridged directly:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("hello from sandbox")

    print("hello from stdout")

At worker boot, `install_worker_bridge()` adds a `logging.Handler` to
the root logger that emits each `LogRecord` on the `/log` SIO
namespace. The host's `RuntimeClient` auto-registers a consumer that
forwards records into the host's own `logging` system, so they appear
in host logs untouched. The worker runtime also captures stdout and sends
each line through the same `/log` stream as `agentix.sandbox.stdout`.

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

from agentix.utils.log._config import configure_logging

__all__ = ["configure_logging", "install_worker_bridge"]


def install_worker_bridge(level: int = logging.NOTSET) -> logging.Handler:
    """Install the bridge handler on the root logger. Idempotent."""
    from agentix.utils.log._bridge import WorkerLogHandler

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
