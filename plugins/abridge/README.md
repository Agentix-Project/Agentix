# abridge

`abridge` is a small mitmproxy-backed bridge for Agentix sandboxes.
It emits protocol-neutral proxy hook events to the host. The first
active response path captures Anthropic Messages requests from an agent
and forwards them to an OpenAI-compatible upstream from the host.

The useful shape is deliberately direct:

```text
agent traffic in sandbox
  -> sandbox-local mitmproxy
  -> sandbox-local HTTP forwarder
  -> Agentix Socket.IO channel
  -> host HookForwarder / OpenAIForwarder
  -> optional upstream API
```

The upstream API key stays on the host. The sandbox only receives a
local Anthropic-shaped URL, so the core protocol is not tied to Docker
host networking.

## Install

```bash
pip install "agentix-bridge[mitm]"
```

## Sandbox usage

Start the proxy inside the sandbox with `RuntimeClient.remote(...)`:

```python
import agentix.bridge.mitm as abridge_mitm

proxy = await client.remote(abridge_mitm.start_proxy)
print(proxy.url)  # pass this as ANTHROPIC_BASE_URL
```

Register the host forwarder before opening the runtime client:

```python
import agentix.bridge.mitm as abridge_mitm
from agentix import RuntimeClient

forwarder = abridge_mitm.OpenAIForwarder(
    base_url="https://example.com/v1",
    api_key="sk-...",
    model="your-openai-compatible-model",
    extra_body={"enable_thinking": True},
)

client = RuntimeClient(runtime_url)
client.register_namespace(forwarder)
async with client as c:
    proxy = await c.remote(abridge_mitm.start_proxy)
    ...
    await c.remote(abridge_mitm.stop_proxy, handle=proxy)
```

For non-LLM traffic, register a plain hook handler:

```python
async def on_proxy_event(event):
    print(event["kind"], event.get("protocol"))
    return {"action": "continue"}

client = RuntimeClient(runtime_url)
client.register_namespace(abridge_mitm.HookForwarder(on_proxy_event))
```

## Raw mitmproxy CLI

The package also exposes the addon directly for local development:

```bash
uv run --extra mitm abridge-mitm --mode wireguard --listen-port 8080
```

When `ABRIDGE_HOOK_URL` is set, captured hook events are sent to
that local hook receiver. Without it, the addon can still call an
OpenAI-compatible upstream directly using:

```bash
OPENAI_BASE_URL=https://example.com/v1 \
OPENAI_API_KEY=sk-... \
OPENAI_MODEL=your-model \
uv run --extra mitm abridge-mitm
```

## Current scope

- Emits protocol-neutral mitmproxy hook events for HTTP, WebSocket,
  TCP, UDP, and DNS.
- Captures HTTP Anthropic Messages calls through mitmproxy.
- Translates Anthropic Messages request bodies to OpenAI Chat
  Completions request bodies.
- Converts buffered OpenAI-compatible responses back to Anthropic JSON
  or Anthropic SSE for clients that requested streaming.
- Leaves true chunk-by-chunk streaming and broader protocol support as
  the next extension points.
