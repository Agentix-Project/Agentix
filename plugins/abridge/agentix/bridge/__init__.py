"""agentix.bridge — tunnel an in-sandbox agent's LLM traffic to the host.

The agent runs *inside* the sandbox, so its LLM calls originate there.
abridge ferries them over the runtime's Socket.IO connection to the host,
which holds the real key and forwards to any OpenAI-compatible endpoint
(OpenAI, OpenRouter, or your own gateway). Every call is captured and
turned into a `/trace` span. The agent code is never touched.

Host side — register the consumer before the first remote call:

```python
from agentix.bridge import OpenAICompatibleClient, InMemoryStore

store = InMemoryStore()
host = OpenAICompatibleClient(
    base_url="https://api.openai.com/v1",   # or your gateway / vLLM / OpenRouter
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",                    # pin the upstream model (optional)
    store=store,
)
sandbox.register_namespace(host)
```

Sandbox side — one remote call runs the agent with the proxy live around
it (`bridged` starts/stops the proxy and points the SDK env at it):

```python
from agentix.bridge import BridgeConfig, bridged
from my_agent import solve                  # an importable agent callable

answer = await sandbox.remote(
    bridged, solve, task, _bridge=BridgeConfig(session_id=sid)
)
```

Afterwards, `store.trajectory(sid)` is the agent-eye text trajectory for
that rollout (token-level data — ids/logprobs — lives in your gateway,
keyed by the same `session_id`).

See `ROADMAP.md` next to this file for planned extensions.
"""

from __future__ import annotations

from .client import OpenAICompatibleClient, UpstreamHook
from .detection import ApiFamily, detect
from .proxy import (
    NAMESPACE,
    RECORD_EVENT,
    REQUEST_EVENT,
    BridgeConfig,
    ProxyHandle,
    bridged,
    export_environ,
    start_proxy,
    stop_proxy,
)
from .storage import (
    CompletionRecord,
    InMemoryStore,
    TokenUsage,
    extract_usage,
    make_record,
)

__version__ = "0.3.0"

__all__ = [
    "NAMESPACE",
    "ApiFamily",
    "BridgeConfig",
    "CompletionRecord",
    "InMemoryStore",
    "OpenAICompatibleClient",
    "ProxyHandle",
    "RECORD_EVENT",
    "REQUEST_EVENT",
    "TokenUsage",
    "UpstreamHook",
    "__version__",
    "bridged",
    "detect",
    "export_environ",
    "extract_usage",
    "make_record",
    "start_proxy",
    "stop_proxy",
]
