# abridge Roadmap

`agentix-bridge` is a shape-blind JSON-POST→SIO tunnel + a host-side
`Proxy` that routes path-named SIO events to `@on(path)`-decorated
handler methods. Three bundled handlers in `agentix.bridge.clients`
cover the OpenAI and Anthropic cases; custom handlers are plain
classes you write yourself.

This roadmap is the design plan for the next layers. Every entry
preserves today's surface:

```python
from agentix.bridge import (
    Proxy,              # host SIO consumer + sandbox tunnel lifecycle
    on,                 # @on(path) — decorator that wires a method to a URL path
    Client,             # marker Protocol for "any class with @on methods"
    Handler,            # type alias for the bound @on method shape
    Request,            # what an @on method receives (path + decoded JSON object)
    ClientResponse,     # one buffered JSON / SSE / byte response
    AbridgeError,       # raise for in-band agent-side errors (carries status_code)
    TunnelHandle,       # what proxy.start(sandbox) yields (url, port)
)

from agentix.bridge.clients import (
    OpenAIClient, AnthropicClient, AnthropicFromOpenAIClient,
    OPENAI_PLACEHOLDER_API_KEY, ANTHROPIC_PLACEHOLDER_API_KEY,
    populate_openai_span, populate_anthropic_span,
)
# `environ(handle)` is an instance method on the two Anthropic-side
# clients; OpenAI agents typically use base_url=/api_key= args instead.
```

## Layered structure (today)

```
sandbox tunnel       ── http://127.0.0.1:<port>/<declared path>
   │  whitelist routes from `Proxy.paths`; JSON POST object only
   ▼  SIO /abridge   event name == URL path; payload == agent body (no envelope)
host Proxy
   │  trigger_event(path) → @on(path) handler (detached task)
   │  bundled clients open their own trace.span(...) and call
   │  populate_*_span for OTel GenAI attrs
   ▼  one buffered ClientResponse (bytes + media_type + status)
```

`Forward` and `Sidecar` add an optional host-side process boundary to this
structure. They do not widen the HTTP contract: `Forward` re-serializes the
decoded JSON object and buffers the full sidecar response. Existing bundled
clients remain valid while sidecar gateways mature.

## Near-term — same package

1. **Real streaming.** Today `AnthropicClient` streams via the SDK
   internally but re-serialises the whole stream as a single SSE blob
   before returning, and `Forward` buffers the complete HTTP response.
   `AnthropicFromOpenAIClient` is non-streaming upstream regardless of
   the agent's `stream=True`. Real streaming =
   split the SIO request into `<path>:open` / `<path>:chunk` /
   `<path>:end`, iterate `chat.completions.stream(...)` on the host,
   forward chunks through the tunnel as they arrive.

2. **Required sidecar integration coverage.** Install a pinned gateway
   binary in CI and exercise sandbox tunnel → SIO → `Forward` → sidecar →
   mock upstream, including HTTP errors and lifecycle cleanup.

3. **`ReplayClient`** under `clients/replay.py`. Wraps a list of
   pre-captured `(request, ClientResponse)` pairs; satisfies any
   `@on(path)` by index. Useful for offline eval reruns, RL buffer
   regression tests, CI-friendly assertions without burning tokens.

4. **Capture API.** Today storage is gone from the core (skipped in
   the recent cleanup). Add `agentix.bridge.capture` — a small hook
   that any handler can call (or a Proxy-level event subscriber) to
   record full `(request, response)` pairs. Lightweight; in-memory
   list with optional `JsonlSink` / `ParquetSink` overlays.

## Medium-term — additional bundled clients

Each new family gets a sibling under `clients/`:

* **`clients.gemini.GeminiClient`** — native Gemini Generative
  Language API.
* **`clients.cohere.CohereClient`** — native Cohere v2 Chat.
* **`clients.bedrock.BedrockClient`** — native Bedrock Converse API.
* **`*FromOpenAIClient` adapters** for each — translates an agent's
  preferred shape to OpenAI on the upstream side.
* **`OpenAIFromAnthropicClient`** — OpenAI agent → native Anthropic
  upstream (reverse direction; useful for testing OpenAI agents
  against Claude).

Each bundled client owns its SDK dep (declared as an optional extra
in `pyproject.toml`) and its `environ(handle)` instance method if the
SDK reads URL/key from env vars (Anthropic-side). OpenAI-side clients
don't ship `environ` — OpenAI agents typically pass `base_url=` and
`api_key=` as constructor args.

## Long-term — training-bridge surface

`polar.gateway`'s pause/resume + completion writer split, when this
package is paired with a training loop:

* **Pause / resume controls.** A `TrainerClient` wrapper that stops
  new generation while weights are being updated, then resumes once
  the backend is ready. Easy as a `@on` decorator around an inner
  client; doesn't need anything from the bridge core.

* **CompletionWriter.** A capture sink that writes directly into
  parquet shards / Kafka / a registered HF dataset, decoupled from
  the in-memory store.

* **Session API.** Today `session_id` lives on the `Client` instance
  (reused across proxies = same session). The trainer-facing Session
  adds pause/resume + persistence handles on top of that grouping key.

## Non-goals

* **Re-implementing mitmproxy.** The tunnel is intentionally a regular
  FastAPI server; SDK base-URL overrides are enough.
* **Owning credentials inside the sandbox.** The sandbox never reads
  the real API key; only the host process holds it.
* **Per-token billing.** Token usage comes from the upstream response
  via `populate_*_span` attrs; cost accounting is downstream.
* **Built-in shape detection.** Proxy stays shape-blind. Each bundled
  client knows its own shape; users writing custom handlers do the
  same. There is no `detect()` and no path-sniffing in the core.
