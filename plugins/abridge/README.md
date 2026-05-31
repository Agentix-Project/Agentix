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

## Usage

The agent runs **inside** the sandbox; `bridged` brings the proxy up
around it (and tears it down after), points the SDK base-URL env vars at
the proxy, and runs sync agents off the event loop so the proxy stays
responsive. The host registers the consumer that forwards to the real
endpoint and captures every call.

```python
import uuid
from agentix.bridge import BridgeConfig, OpenAICompatibleClient, InMemoryStore, bridged
from my_agent import solve   # an importable agent callable (NOT a __main__ function)

store = InMemoryStore()
host = OpenAICompatibleClient(
    base_url="https://api.openai.com/v1",   # OpenAI / OpenRouter / vLLM / your gateway
    api_key="sk-...",
    model="gpt-4o-mini",                    # pin the upstream model (optional)
    store=store,
)

session_id = uuid.uuid4().hex
async with provider.session(SandboxConfig(...)) as sandbox:
    sandbox.register_namespace(host)        # host half of /abridge — before first remote
    answer = await sandbox.remote(
        bridged, solve, task, _bridge=BridgeConfig(session_id=session_id)
    )

# Agent-eye text trajectory for this rollout (token-level data lives in
# your gateway, keyed by the same session_id):
for rec in store.trajectory(session_id):
    print(rec.request_id, rec.family.value, rec.usage.total_tokens)
```

The agent (`solve`) is pristine — it just constructs the Anthropic /
OpenAI SDK and calls it; `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` already
point at the in-sandbox proxy. It must be an **importable** function
(`module::qualname`), not defined in the `__main__` script, so it
resolves inside the sandbox.

`base_url` is just an OpenAI-compatible endpoint — OpenAI, Azure, vLLM,
OpenRouter, or a separate gateway (e.g. an SGLang RL wrapper). abridge
forwards `x-session-id` / `x-request-id` headers so such a gateway can
group its own token-level trajectory by session.

### Tracing

Each call becomes a `/trace` span tagged per OpenTelemetry GenAI
conventions (`gen_ai.request.model`, `gen_ai.usage.*`, tool calls as a
span event). abridge only *produces* spans; register a `trace.Processor`
to export them to any OTel backend.

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
