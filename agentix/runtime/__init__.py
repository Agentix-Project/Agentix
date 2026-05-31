"""Runtime subpackage — split into three sides.

  * `agentix.runtime.shared`  — wire types, framing, codec, event-name
    constants. Both client and server depend on this; nothing here
    depends on `client/` or `server/`.
  * `agentix.runtime.client`  — orchestrator-side `RuntimeClient`
    (HTTP health checks; Socket.IO remote calls).
  * `agentix.runtime.server`  — sandbox-side: FastAPI app, Socket.IO
    server, the `RuntimeWorkerClient`, and the `worker` subprocess
    (`python -m agentix.runtime.server.worker`).

Importing this top-level package does NOT eagerly import `client` or
`server` — that would widen the import graph unnecessarily when other
modules pull wire types from `agentix.runtime.shared.models`.
Reach for the leaf you need explicitly, e.g.
`from agentix.runtime.client import RuntimeClient`, or use the
top-level re-exports on `agentix`.

The bundle's runtime contract — runtime paths, env vars, and the
entry point every backend execs — lives in `agentix.runtime.shared.env`
(alongside the other cross-side primitives like `MAX_MESSAGE_BYTES` and
`RemoteCallable`). It's re-exported here so
`from agentix.runtime import BUNDLE_RUNTIME_ENTRYPOINT` keeps working
for provider plugins; the canonical home is `shared/env.py`.
"""

from __future__ import annotations

from agentix.runtime.shared.env import (
    AGENTIX_ADDED_LD_LIBRARY_PATH,
    AGENTIX_ADDED_PATH,
    BIND_HOST_ENV,
    BIND_PORT_ENV,
    BUNDLE_NIX_ROOT,
    BUNDLE_RUNTIME_BASH,
    BUNDLE_RUNTIME_BIN,
    BUNDLE_RUNTIME_ENTRYPOINT,
    BUNDLE_RUNTIME_ENV,
    BUNDLE_RUNTIME_INCLUDE,
    BUNDLE_RUNTIME_LIB,
    BUNDLE_RUNTIME_PATH_ENTRIES,
    BUNDLE_RUNTIME_PKGCONFIG_DIRS,
    BUNDLE_RUNTIME_ROOT,
    BUNDLE_RUNTIME_VENV,
    BUNDLE_RUNTIME_VENV_BIN,
    get_env_without_agentix,
)

__all__ = [
    "AGENTIX_ADDED_LD_LIBRARY_PATH",
    "AGENTIX_ADDED_PATH",
    "BIND_HOST_ENV",
    "BIND_PORT_ENV",
    "BUNDLE_NIX_ROOT",
    "BUNDLE_RUNTIME_BASH",
    "BUNDLE_RUNTIME_BIN",
    "BUNDLE_RUNTIME_ENTRYPOINT",
    "BUNDLE_RUNTIME_ENV",
    "BUNDLE_RUNTIME_INCLUDE",
    "BUNDLE_RUNTIME_LIB",
    "BUNDLE_RUNTIME_PATH_ENTRIES",
    "BUNDLE_RUNTIME_PKGCONFIG_DIRS",
    "BUNDLE_RUNTIME_ROOT",
    "BUNDLE_RUNTIME_VENV",
    "BUNDLE_RUNTIME_VENV_BIN",
    "get_env_without_agentix",
]
