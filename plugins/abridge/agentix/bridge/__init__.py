"""agentix.bridge — tunnel an in-sandbox agent's LLM traffic to the host.

The agent runs *inside* the sandbox; its LLM calls hit a tiny HTTP
tunnel on `127.0.0.1` that ferries each request over Agentix's Socket.IO
connection to the host. On the host, a `Proxy` routes by URL path to
`@on(path)`-decorated handler methods. The handler does whatever the
call needs (POST upstream, replay, mock, route by body) and returns a
`ClientResponse`; the bridge ferries the bytes back to the agent.

```python
from agentix.bridge import Proxy
from agentix.bridge.clients import AnthropicFromOpenAIClient, anthropic_env_for

proxy = Proxy(AnthropicFromOpenAIClient(
    base_url="https://api.openai.com/v1", api_key=..., model="gpt-4o",
))

async with provider.session(cfg) as sandbox:
    async with proxy.session(sandbox) as handle:
        await sandbox.remote(agent, env=anthropic_env_for(handle))
```

Custom handlers are plain classes with `@on(path)` methods — no
Protocol, no base class, no registration. Multiple handlers compose
via Python multiple inheritance (mixins) or by passing several to
`Proxy(*handlers)`. Two handlers must not register the same path.

`agentix.bridge.clients` ships the three standard handlers
(`OpenAIClient`, `AnthropicClient`, `AnthropicFromOpenAIClient`), each
built on the corresponding provider SDK. Pull from there as building
blocks; abridge core stays shape-blind.
"""

from __future__ import annotations

from .forward import Forward, SessionForward
from .proxy import (
    NAMESPACE,
    AbridgeError,
    Client,
    ClientResponse,
    Handler,
    Proxy,
    Request,
    TunnelHandle,
    on,
)
from .sidecar import Command, Sidecar, SidecarError

__version__ = "0.5.0"

__all__ = [
    "AbridgeError",
    "Client",
    "ClientResponse",
    "Command",
    "Forward",
    "Handler",
    "NAMESPACE",
    "Proxy",
    "Request",
    "Sidecar",
    "SidecarError",
    "TunnelHandle",
    "__version__",
    "on",
]
