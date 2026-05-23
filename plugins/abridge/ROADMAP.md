# abridge Roadmap

`agentix-bridge` is the Agentix LLM gateway. Today it captures
Anthropic Messages and OpenAI Chat Completions traffic from inside a
sandbox, ferries the request to the host, and returns the response.
A `CompletionRecord` is buffered on the host for every call so the
caller can attach LLM traces to a rollout, score a trajectory, or feed
an RL buffer.

This roadmap is the design plan for the next layers we expect to add.
Every entry should preserve today's public surface:

```python
from agentix.bridge import (
    OpenAICompatibleClient,   # host SIO consumer
    start_proxy, stop_proxy,  # sandbox lifecycle
    export_environ,           # env-var hand-off for agent SDKs
    InMemoryStore,            # capture sink
)
```

Anything in this document is opt-in until shipped — until then,
agents can keep using the OpenAI-compatible host client unchanged.

## Layered structure (today)

```
sandbox proxy        ── http://127.0.0.1:<port>
   │  detect() ────────── family classification (OpenAI / Anthropic / …)
   │  transform/ ─────── per-family request <-> OpenAI shape
   │  storage ────────── CompletionRecord + bounded InMemoryStore
   ▼  SIO /abridge
host client
   │  OpenAICompatibleClient  ── upstream POST /chat/completions
```

This is intentionally a subset of `polar.gateway`'s gateway tree:
`detection.py`, `transform/`, `proxy.py`, `storage.py` map 1:1, while
`dispatcher.py`, `session.py`, `completion_writer.py`, and the
pause/resume training-bridge controls land in this repo under
`agentix/gateway/` (separate package, see the parent ROADMAP).

## Near-term — same package, no new dependencies

1. **Streaming proxy.** Today the sandbox always asks the upstream
   for `stream=False` and synthesises an SSE blob from the final
   message when the agent asked for streaming. Move to true streaming
   by:
   * splitting the SIO request into `llm_call:open` / `llm_call:chunk`
     / `llm_call:end`,
   * iterating over `httpx.AsyncClient.stream(...)` on the host,
   * re-encoding Anthropic SSE chunks block-by-block instead of one
     post-hoc envelope.

2. **Per-call upstream selection.** `OpenAICompatibleClient` is one
   model with one key. Add a `route(payload) -> RouteSpec` hook so a
   single host process can fan out across providers (OpenAI vs Azure
   vs a local vLLM) based on the requested model name.

3. **Replay mode.** A `ReplayClient(records)` host-side namespace
   that satisfies `llm_call` from a previously captured
   `InMemoryStore` snapshot — useful for offline eval reruns, RL
   buffer regression tests, and CI-friendly assertions of agent
   behaviour without burning tokens.

4. **Jsonl / parquet sink alongside the buffer.** `InMemoryStore` is
   the source of truth; a `Persistent...Sink(store, path)` writes
   each record on `add()` so a long agent run survives a host
   process crash. Keep the buffer; the sink is a tee.

5. **Tracing integration (task 3 in the parent batch).** Open a
   `trace.span("llm.request", model=…)` per request, attach the
   `record_id`, emit `add_event("first_chunk")` for streaming, and
   close on completion. Spans cross the sandbox/host boundary already
   via `/trace`, so the captured calls show up next to the rest of
   the agent's work in any OTel-compatible viewer (task 4).

## Medium-term — additional API families

Each family gets its own transform module and a recogniser in
`detection.py`:

* **Gemini Generative Language API** — Gemini's
  `/v1/models/<id>:generateContent` body is structurally close
  enough to OpenAI that the converter is just a renaming pass.
* **Cohere v2 Chat** — adopts the OpenAI shape mostly; a thin shim.
* **Bedrock Converse API** — distinct schema, distinct streaming
  envelope.
* **Reasoning-heavy responses (o1, o3, …)** — propagate
  `reasoning_effort`, `reasoning_content`, and reasoning-token counts
  through both the proxy and the `CompletionRecord`.

When a family appears here, the host gets an optional native client
mirroring `OpenAICompatibleClient` (e.g. `BedrockClient`). The proxy
keeps routing through "OpenAI body on the wire" by default, but a
caller who wants to keep the native shape can opt into the family-
specific host client.

## Long-term — training-bridge surface

Adopt `polar.gateway`'s pause/resume + completion writer split when
this package is paired with a training loop:

* **Pause / resume controls.** The host exposes `pause()` /
  `resume()` so a trainer can stop new generation while weights are
  being updated, then resume once the backend is ready. The sandbox
  proxy stalls open requests at the `llm_call:open` step rather than
  failing.

* **CompletionWriter.** Move record persistence behind a writer
  interface so high-throughput RL rollouts can write directly into
  parquet shards / Kafka / a registered HF dataset.

* **Session API.** A `Session` object groups multiple LLM calls under
  one logical rollout so persistence and tracing have a stable key
  beyond `request_id`. Maps onto `polar.gateway.session`.

The `agentix/gateway/` package (parent ROADMAP, task 7) is the home
for the trainer-facing pieces; abridge stays focused on the
proxy/translate/capture surface and reuses the gateway's session
model when both packages are installed.

## Non-goals

* Re-implementing mitmproxy. The proxy intentionally is a regular
  FastAPI server; SDK base-URL overrides are enough.
* Owning credentials inside the sandbox. The sandbox proxy never
  reads the real API key; only the host process holds it.
* Per-token billing. `CompletionRecord.usage` carries the upstream's
  reported token counts; the host or a downstream consumer is
  responsible for any cost accounting on top of those.
