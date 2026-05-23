"""agentix.utils — side-channel observability primitives.

Two sub-packages live here:

  - `agentix.utils.log`   — stdlib `logging` bridge (sandbox -> host).
  - `agentix.utils.trace` — OTel-style Trace + Span + SpanEvent.

Both are *side channels* relative to `c.remote(fn, ...)`: they ride
dedicated Socket.IO namespaces (`/log`, `/trace`) and are not part of
the RPC result path.
"""

from __future__ import annotations

import pkgutil

# Mirror what `agentix/__init__.py` does: make `agentix.utils` a
# namespace-aware regular package so plugins (e.g. `agentix-trace-otel`
# shipping `agentix.utils.trace.otel`) can contribute siblings under
# the utils tree. Without this, `agentix.utils.__path__` only includes
# the core path and the plugin's `agentix/utils/trace/otel.py` is
# invisible to the import system.
__path__ = pkgutil.extend_path(__path__, __name__)

from agentix.utils import log, trace  # noqa: E402

__all__ = ["log", "trace"]
