"""Agentix gateway — sandbox sessions, dispatch, capture, rollouts.

A small HTTP service that

  * accepts session dispatches from a coordinator (a rollout server,
    a notebook, a CI job — anything that wants to drive sandboxed
    agent runs at scale);
  * provisions a fresh Agentix sandbox per session, runs the agent
    callable inside it via `RuntimeClient.remote(...)`;
  * captures every LLM call through the abridge proxy as a
    `CompletionRecord`;
  * builds a `SessionResult` (status, completion records, trajectory,
    timing, error) and calls a result callback (or stores the result
    for later poll);
  * supports `pause()` / `resume()` so a training bridge can stop new
    generation while weights are being updated.

Module layout:

```
agentix.gateway/
├── server.py            FastAPI app + REST endpoints
├── node.py              Gateway node lifecycle (heartbeat to coordinator)
├── dispatcher.py        Stage orchestration: INIT -> READY -> RUNNING -> POSTRUN
├── session.py           Session state, status accounting, result envelope
├── storage.py           In-process record / session stores
└── completion_writer.py Pluggable record sink (jsonl / parquet / kafka / ...)
```

Gateway-side LLM proxying (capture / translate / forward) is the
abridge plugin's job (`agentix.bridge`). Plug an
`OpenAICompatibleClient` into a `GatewayNode` via
`host_namespace_factory` and per-session captured records flow into
the gateway's `RecordStore` automatically.

Public surface for callers:

```python
from agentix.gateway import GatewayNode, Session, SessionSpec, Dispatcher

node = GatewayNode(
    deployment=DockerDeployment(),
    upstream=OpenAICompatibleClient(...),
)
async with node.serve(port=8080):
    ...
```

This module is in active design — the public API may change as we
wire it into the cookbook examples. See `agentix/gateway/ROADMAP.md`
for the planned trajectory.
"""

from __future__ import annotations

from agentix.gateway.dispatcher import Dispatcher, DispatchStage
from agentix.gateway.node import GatewayNode, NodeConfig
from agentix.gateway.server import build_app
from agentix.gateway.session import (
    Session,
    SessionResult,
    SessionSpec,
    SessionStatus,
)
from agentix.gateway.storage import RecordStore, SessionStore

__all__ = [
    "Dispatcher",
    "DispatchStage",
    "GatewayNode",
    "NodeConfig",
    "RecordStore",
    "Session",
    "SessionResult",
    "SessionSpec",
    "SessionStatus",
    "SessionStore",
    "build_app",
]
