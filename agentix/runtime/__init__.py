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

Bundle contract
---------------
These constants describe what every bundle produced by `agentix build`
exposes to the outside world — the bind-mount layout, the entry point
deployment backends exec, and the env var the bundle reads to choose
its listen port. They live here because they're part of the *runtime's*
external contract, not any one deployment backend's convention; every
backend imports the same names instead of hardcoding the paths.

Rename / move the bundle entry point in one place, not N.
"""

from __future__ import annotations

BUNDLE_NIX_ROOT = "/nix"
"""Where the bundle's `/nix` tree is bind-mounted inside the task container."""

BUNDLE_RUNTIME_ROOT = "/nix/runtime"
"""The runtime sub-tree under the bind-mounted bundle root."""

BUNDLE_RUNTIME_ENTRYPOINT = "/nix/runtime/bootstrap.sh"
"""Path inside the bundle that deployment backends exec as the
container entry point. The script preps Nix-managed runtime PATHs and
launches uvicorn against `agentix.runtime.server.app:app`. Backends
should never have to know what the script does — only that it exists
at this path."""

BIND_PORT_ENV = "AGENTIX_BIND_PORT"
"""Env var the bundle's bootstrap script reads to choose its listen
port. Backends pick a free host port and pass it via this name."""


__all__ = [
    "BIND_PORT_ENV",
    "BUNDLE_NIX_ROOT",
    "BUNDLE_RUNTIME_ENTRYPOINT",
    "BUNDLE_RUNTIME_ROOT",
]
