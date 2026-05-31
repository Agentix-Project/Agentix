"""agentix.bridge — tunnel an in-sandbox agent's LLM traffic to the host.

The agent runs *inside* the sandbox, so its LLM calls originate there.
abridge ferries them over the runtime's Socket.IO connection to the host,
which holds the real key and forwards to any OpenAI-compatible endpoint
(OpenAI, OpenRouter, or your own gateway). Every call is captured and
turned into a `/trace` span. The agent code is never touched — you just
point its `base_url` at the bridge's in-sandbox service URL.

```python
from agentix.bridge import Bridge

bridge = Bridge(
    base_url="https://api.openai.com/v1",   # or your gateway / vLLM / OpenRouter
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",                    # pin the upstream model (optional)
)
async with provider.session(cfg) as sandbox:
    await bridge.start_proxy(sandbox, family="anthropic")   # registers + starts the proxy
    # the agent runs with a plain remote; point its base_url at the service:
    result = await sandbox.remote(agent_run, AgentArgs(base_url=bridge.get_base_url(), ...))

for rec in bridge.store.trajectory(bridge.session_id):
    print(rec.request_id, rec.family.value, rec.usage.total_tokens)
```

The token-level trajectory (ids/logprobs) lives in your gateway, keyed by
the same `session_id`. See `ROADMAP.md` for planned extensions.
"""

from __future__ import annotations

from .client import Bridge, UpstreamHook
from .detection import ApiFamily, detect
from .proxy import (
    NAMESPACE,
    RECORD_EVENT,
    REQUEST_EVENT,
    ProxyHandle,
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
    "Bridge",
    "CompletionRecord",
    "InMemoryStore",
    "ProxyHandle",
    "RECORD_EVENT",
    "REQUEST_EVENT",
    "TokenUsage",
    "UpstreamHook",
    "__version__",
    "detect",
    "export_environ",
    "extract_usage",
    "make_record",
    "start_proxy",
    "stop_proxy",
]
