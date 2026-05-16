"""Sandbox-side runtime server.

Composes FastAPI (for HTTP RPC + LLM proxy) and Socket.IO (for streams,
bidi, and log subscription) into the ASGI app uvicorn runs. Routes
every dispatch to a per-namespace worker subprocess via the
`NamespaceMultiplexer`.

Submodules:
  - `app`         — FastAPI app, lifespan, /_remote unary dispatch
  - `sio`         — Socket.IO server + event handlers + log forwarding
  - `llm_proxy`   — reverse-proxy `/_llm/<provider>/<path>` to upstream LLM APIs
  - `trace_bridge` — pipes `agentix.trace.emit(...)` to the Socket.IO `trace` room

Shell exec and file I/O ship as the `bash` and `files` primitive
namespaces under `primitives/`. Invoke via `c.remote(Bash.run, ...)` /
`c.remote(Files.upload, ...)`.
"""

from agentix.runtime.server.app import (
    _multiplexer,
    app,
    main,
)

# `multiplexer` alias for tests that want to introspect or register
# in-process namespaces against the live runtime.
multiplexer = _multiplexer

__all__ = [
    "app",
    "main",
    "multiplexer",
]
