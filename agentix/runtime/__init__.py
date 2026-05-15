"""Runtime subpackage.

Split into:
  - `agentix.runtime.models`  — wire types shared by both transports
  - `agentix.runtime.client`  — orchestrator-side RuntimeClient
  - `agentix.runtime.server`  — sandbox-side FastAPI + Socket.IO app

Importing this top-level package does NOT eagerly import `client` or
`server` — that would create a circular path through `agentix.dispatch`
when other modules pull wire types from `agentix.runtime.models`. Reach
for the leaf you need explicitly, e.g. `from agentix.runtime.client
import RuntimeClient`, or use the top-level re-exports on `agentix`.
"""
