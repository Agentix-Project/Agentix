"""Primitives shared across the runtime — and beyond.

The runtime is split three ways:

  * `agentix.runtime.shared`  — wire types, framing, codec, and the
    bundle's runtime contract. Imported by both sides; also
    imported from `agentix.cli`, `agentix.utils`, and
    `agentix.provider`. No imports back into `client/` or `server/`.
  * `agentix.runtime.client`  — orchestrator-side `RuntimeClient`.
  * `agentix.runtime.server`  — sandbox-side FastAPI app, Socket.IO
    server, and worker subprocess.

Submodules in this package:

  - `callables` — `RemoteCallable` import-path encoding
  - `idents`    — branded NewType ids on the wire (`CallId`)
  - `codec`     — plain msgpack pack/unpack (no ext types)
  - `framing`   — length-prefixed msgpack framing for worker stdio
  - `models`    — pydantic wire types (`RemoteRequest`, `RemoteResponse`, …)
  - `env`       — bundle runtime contract: runtime paths, env vars,
                  `AGENTIX_ADDED_*` tracking + `get_env_without_agentix`
"""

from __future__ import annotations

# Maximum size of a single Socket.IO message (one `c.remote` payload or
# one plugin-namespace event). The default Engine.IO / `websockets`
# cap is 1 MB — far too small: a `c.remote` carrying a pickled object
# graph, or an `abridge` event carrying an LLM request body (system
# prompt + dozens of tool schemas + a growing conversation), routinely
# exceeds it, and the websocket is then killed mid-call. 256 MiB is a
# generous ceiling that still bounds a runaway payload.
#
# Applied in three places that must agree: the Socket.IO server
# (`max_http_buffer_size`), the Socket.IO client (`websocket_extra_
# options.max_size`), and uvicorn's websocket impl (`ws_max_size`).
MAX_MESSAGE_BYTES = 256 * 1024 * 1024

__all__ = ["MAX_MESSAGE_BYTES"]
