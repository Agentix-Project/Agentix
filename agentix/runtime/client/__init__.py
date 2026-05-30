"""Orchestrator-side client for the agentix runtime.

Public surface:
  - `RuntimeClient` — connects to a running sandbox, drives remote calls
    over Socket.IO, and uses HTTP only for health checks.
  - `RemoteCallError` — raised when a remote impl returns a non-ok response.
  - `CallTimeout` / `RuntimeUnreachable` / `WorkerExited` — the typed
    failure vocabulary a caller branches on (deadline exceeded, server
    unreachable, worker subprocess died / OOM).

Implementation lives in `agentix.runtime.client.client`; this package's
`__init__.py` re-exports the public names so the historic import path
`from agentix.runtime.client import RuntimeClient` keeps working.
"""

from agentix.runtime.client.client import (
    CallTimeout,
    RemoteCallError,
    RuntimeClient,
    RuntimeUnreachable,
    WorkerExited,
)

__all__ = [
    "CallTimeout",
    "RemoteCallError",
    "RuntimeClient",
    "RuntimeUnreachable",
    "WorkerExited",
]
