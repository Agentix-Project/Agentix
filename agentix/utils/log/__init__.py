"""agentix.utils.log — sandbox stdout/stderr captured and ferried to the host.

Workers need no logging API. The runtime captures the worker's stdout and
stderr (stdlib `logging` writes to stderr, so it is captured too) and streams
each line best-effort on the `/log` namespace, replayed on the host under
`agentix.sandbox.{stdout,stderr}`. A durable copy is written to a sandbox-side
file.

    import logging
    logging.getLogger(__name__).info("hello from sandbox")  # -> host logs
    print("hello from stdout")                              # -> host logs

`configure_logging` sets up local stdlib logging for the host, runtime, and
worker processes (level / format / context from `AGENTIX_LOG_*` env vars).
"""

from __future__ import annotations

from agentix.utils.log._config import configure_logging

__all__ = ["configure_logging"]
