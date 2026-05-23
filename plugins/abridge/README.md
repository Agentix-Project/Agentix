# abridge

`agentix-bridge` is the Agentix LLM gateway. It runs a small HTTP
proxy inside the sandbox that captures Anthropic Messages and OpenAI
Chat Completions traffic, ferries each request to the host over the
existing Agentix Socket.IO connection, and returns the response.
Every captured call is materialised as a `CompletionRecord` on the
host so a caller can attach it to a rollout, feed an RL buffer, or
score a trajectory.

```text
agent in sandbox
  -> http://127.0.0.1:<port>  (sandbox proxy: detect + translate + capture)
  -> Agentix /abridge SIO namespace
  -> host OpenAICompatibleClient (POST /chat/completions upstream)
  <- response (translated back to Anthropic if the agent asked for it)
```

The real upstream API key lives only on the host. Inside the sandbox,
agents see a regular HTTP service on `127.0.0.1`; no TLS interception,
no mitmproxy subprocess.

## Install

```bash
pip install agentix-bridge
```

## Sandbox usage

```python
import os
import agentix.bridge.proxy as proxy
from agentix import RuntimeClient

async with RuntimeClient(runtime_url) as c:
    handle = await c.remote(proxy.start_proxy)
    env = await c.remote(proxy.export_environ, handle=handle)
    # ... pass `env` to whatever agent harness you run inside the
    # sandbox: it picks up ANTHROPIC_BASE_URL / OPENAI_BASE_URL and
    # talks to 127.0.0.1 instead of the real provider.
    ...
    await c.remote(proxy.stop_proxy, handle=handle)
```

## Host usage

Register an `OpenAICompatibleClient` with the runtime client before
opening it, so the host is listening for `llm_call` events from the
sandbox proxy:

```python
from agentix import RuntimeClient
from agentix.bridge import OpenAICompatibleClient, InMemoryStore

store = InMemoryStore()
host = OpenAICompatibleClient(
    base_url="https://example.com/v1",
    api_key="sk-...",
    model="your-openai-compatible-model",
    store=store,
)

client = RuntimeClient(runtime_url)
client.register_namespace(host)
async with client as c:
    # ... drive the agent ...
    pass

for record in store.snapshot():
    print(record.request_id, record.family, record.usage.total_tokens)
```

`OpenAICompatibleClient` works for any OpenAI-compatible endpoint —
OpenAI itself, Azure OpenAI, vLLM, Together, Anyscale, a local
`llama.cpp` server.

## Modules

* `agentix.bridge.proxy` — sandbox HTTP server + SIO namespace.
* `agentix.bridge.client` — host `OpenAICompatibleClient` + record sink.
* `agentix.bridge.detection` — request API-family classification.
* `agentix.bridge.transform.anthropic` — Anthropic <-> OpenAI shape
  converters (request + response + SSE renderer + count-tokens
  estimator). New families live as siblings under
  `agentix.bridge.transform/`.
* `agentix.bridge.storage` — `CompletionRecord`, `TokenUsage`,
  `InMemoryStore`. Persistence sinks belong to consumers.

## What's next

See [ROADMAP.md](ROADMAP.md) for the medium-term plan: real streaming
proxy, per-call upstream routing, replay mode, tracing integration,
Gemini / Bedrock / Cohere shape converters, and the eventual
training-bridge pause/resume surface (which lands in
`agentix/gateway/`, not here).
