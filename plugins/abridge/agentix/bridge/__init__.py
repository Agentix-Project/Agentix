"""agentix.bridge — sandbox LLM proxy + host OpenAI-compatible client.

Sandbox side:

```python
from agentix.bridge import start_proxy, stop_proxy, export_environ

handle = await start_proxy()
os.environ.update(export_environ(handle))
# ... run an agent harness; it now hits 127.0.0.1 instead of the real API.
await stop_proxy(handle)
```

Host side:

```python
from agentix.bridge import OpenAICompatibleClient

client = OpenAICompatibleClient(
    base_url="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
)
runtime_client.register_namespace(client)
# Records: client.store.snapshot() after the run.
```

See `ROADMAP.md` next to this file for planned extensions (streaming
proxy, host Anthropic client, training-bridge pause/resume, etc.).
"""

from __future__ import annotations

from .client import OpenAICompatibleClient, UpstreamHook
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
from .transform import (
    AnthropicCountTokens,
    anthropic_messages_to_openai,
    anthropic_sse,
    count_anthropic_tokens,
    openai_to_anthropic_messages,
)

__version__ = "0.3.0"

__all__ = [
    "ApiFamily",
    "AnthropicCountTokens",
    "CompletionRecord",
    "InMemoryStore",
    "NAMESPACE",
    "OpenAICompatibleClient",
    "ProxyHandle",
    "RECORD_EVENT",
    "REQUEST_EVENT",
    "TokenUsage",
    "UpstreamHook",
    "__version__",
    "anthropic_messages_to_openai",
    "anthropic_sse",
    "count_anthropic_tokens",
    "detect",
    "export_environ",
    "extract_usage",
    "make_record",
    "openai_to_anthropic_messages",
    "start_proxy",
    "stop_proxy",
]
