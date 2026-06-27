# abridge

`agentix-bridge` is the Agentix HTTP tunnel. Inside the sandbox it runs
a tiny HTTP server on `127.0.0.1` that catches your agent's outbound
JSON POST calls and ferries each decoded object over Agentix's Socket.IO connection to the
host. On the host, a `Proxy` routes by URL path to `@on(path)`-decorated
handler methods you supply. The handler decides what happens (POST
upstream, translate shapes, replay, mock) and returns a
`ClientResponse`; the bridge returns its fully buffered body to the agent.

```text
agent in sandbox
  -> http://127.0.0.1:<port>/<declared path>    (sandbox tunnel, JSON POST)
  -> Agentix /abridge SIO namespace             (SIO event name == URL path)
  -> host Proxy → your @on(path) method         (your code)
  <- ClientResponse (bytes + media_type)
```

abridge's core is **shape-blind**. It does not know Anthropic from OpenAI
and does not predefine message fields, but its current transport contract
is deliberately narrower than a generic HTTP proxy: declared POST routes,
JSON object bodies, and one buffered response per request. Request headers,
query parameters, non-JSON bodies, and incremental response chunks are not
forwarded today. Bundled handlers in `agentix.bridge.clients` cover OpenAI
and Anthropic; custom JSON protocols can use the same machinery.

## Install

```bash
pip install agentix-bridge[openai]      # OpenAIClient + AnthropicFromOpenAIClient
pip install agentix-bridge[anthropic]   # AnthropicClient
pip install agentix-bridge[all]         # both SDKs
pip install agentix-bridge              # core only; bring your own handler classes
```

The provider SDKs (`openai`, `anthropic`) are **optional extras** —
they're only needed for the bundled clients that use them. Custom
handlers can use raw httpx, mocks, or anything else without pulling
either SDK.

## Five-minute usage

### Anthropic agent → OpenAI upstream (Claude Code, Anthropic SDK)

```python
from agentix.bridge import Proxy
from agentix.bridge.clients import AnthropicFromOpenAIClient

client = AnthropicFromOpenAIClient(
    base_url="https://api.openai.com/v1",   # OpenAI / OpenRouter / vLLM / your gateway
    api_key="sk-...",
    upstream_model="gpt-4o",                # the agent keeps sending claude-* model ids
)
proxy = Proxy(client)

async with provider.session(cfg) as sandbox:
    async with proxy.session(sandbox) as handle:
        await sandbox.remote(agent, env=client.environ(handle))
```

`client.environ(handle)` returns
`{"ANTHROPIC_BASE_URL": ..., "ANTHROPIC_API_KEY": "<placeholder>"}`.
The placeholder key matches Anthropic's real key format
(`sk-ant-api03-...`) so SDK-side validation passes; the real upstream
credential lives only on the host (the wrapped OpenAI client).

### OpenAI agent → OpenAI upstream

```python
from agentix.bridge import Proxy
from agentix.bridge.clients import OpenAIClient
from agentix.bridge.clients.openai import PLACEHOLDER_API_KEY

client = OpenAIClient(base_url=..., api_key=..., model="gpt-4o")
proxy = Proxy(client)

async with proxy.session(sandbox) as handle:
    await sandbox.remote(agent, base_url=f"{handle.url}/v1", api_key=PLACEHOLDER_API_KEY)
```

`OpenAIClient` doesn't ship an `environ(handle)` helper — most OpenAI
SDK callers construct the client with explicit `base_url=`/`api_key=`
arguments rather than reading env vars.

### Anthropic agent → native Anthropic upstream

```python
from agentix.bridge import Proxy
from agentix.bridge.clients import AnthropicClient

client = AnthropicClient(api_key="sk-ant-...")   # default base_url = api.anthropic.com
proxy = Proxy(client)

async with proxy.session(sandbox) as handle:
    await sandbox.remote(agent, env=client.environ(handle))
```

### Forward through a host-side sidecar

`Forward` keeps protocol-specific translation outside abridge. `Sidecar`
owns a local process; an external service URL can be passed directly when
another system owns its lifecycle.

```python
from agentix.bridge import Forward, Proxy, Sidecar

async with Sidecar(
    command=["my-sidecar", "--listen", "{host}:{port}"],
    health_path="/healthz",
) as sidecar_url:
    proxy = Proxy(Forward(sidecar_url, paths=["/v1/messages"]))
    async with proxy.session(sandbox) as handle:
        await sandbox.remote(agent, base_url=handle.url)
```

This path forwards JSON bodies and buffers the complete sidecar response.
An SSE payload is preserved as `text/event-stream`, but chunks do not reach
the sandbox incrementally yet.

## Writing your own handler

Any class with `@on(path)`-decorated methods works. No base class to
inherit, no Protocol to satisfy.

```python
from agentix.bridge import Proxy, Request, ClientResponse, AbridgeError, on

class MyClient:
    @on("/v1/messages")
    async def messages(self, request: Request) -> ClientResponse:
        # Inspect / route / mock / replay — whatever you want.
        if some_condition:
            raise AbridgeError("nope", status_code=503)  # in-band error to the agent
        return ClientResponse.json({"id": "...", "content": [...], ...})

proxy = Proxy(MyClient())
```

Common patterns:

- **Per-call routing.** Inspect `request.body["model"]` and dispatch to
  different upstreams.
- **Replay.** Wrap a list of pre-captured responses; return the next
  one on each call.
- **RL trainer hook.** Pause/resume inside `messages()` while weights
  swap; record logprobs from the upstream response.
- **MCP / custom RPC.** One `@on("/mcp")` that dispatches on
  `request.body["method"]` — abridge doesn't care about the protocol,
  just the URL path.
- **Test doubles.** Return canned dicts; no upstream needed.

## Composing multiple handlers

Two ways to combine handler sets in one Proxy. Pick whichever fits.

### Variadic constructor (composition)

```python
proxy = Proxy(OpenAIClient(...), MyCustomTool(...))
```

The Proxy walks each client for `@on(...)` methods. Two clients
registering the same path is a construction-time error.

### Mixin (multiple inheritance)

```python
class WebFetchTool:
    @on("/v1/webfetch")
    async def fetch(self, request): ...

class MyClient(OpenAIClient, WebFetchTool):
    pass

proxy = Proxy(MyClient(base_url=..., api_key=...))
```

Mixins must register disjoint paths. They don't call each other — each
`@on(...)` method is independently routed.

## Observability

abridge's `Proxy` and tunnel do **no tracing themselves** — caller-side
`trace.span(...)` doesn't propagate across the HTTP/SIO boundary, so
each bundled client opens its own `trace.span(...)` inside its `@on`
method (named like `openai chat <model>` / `anthropic messages
<model>`). Inside that span the client calls `populate_openai_span` /
`populate_anthropic_span` from `agentix.bridge.clients` to stamp OTel
GenAI attrs (`gen_ai.request.model`, `gen_ai.usage.*`, prompt /
completion content, tool-call names).

Custom handlers can do the same — open a span, call the populate
helpers (or set attrs directly with `trace.get_current_span()`).

`@on(path)` itself wraps every invocation with DEBUG entry + INFO
completion logs (elapsed-ms, status code). Wire-level errors come from
`Proxy._dispatch_request` at WARNING / EXCEPTION level. Register a
`trace.Processor` (e.g. `agentix.plugins.trace-otel`) to export to
LangSmith / Langfuse / Datadog / any OTel backend.

## Module layout

```
agentix/bridge/
├── proxy.py                       # Proxy + @on + sandbox tunnel + wire types
├── forward.py                     # JSON POST forwarding to a host-side service
├── sidecar.py                     # local process lifecycle + health supervision
├── sidecars.py                    # presets for external sidecar binaries
└── clients/                       # bundled handler implementations
    ├── openai.py                  # OpenAIClient (openai SDK) + PLACEHOLDER_API_KEY
    ├── anthropic.py               # AnthropicClient (anthropic SDK) + environ() + PLACEHOLDER_API_KEY
    ├── anthropic_from_openai.py   # AnthropicFromOpenAIClient (openai SDK + translation) + environ()
    ├── _genai_span.py             # populate_openai_span / populate_anthropic_span
    └── _anthropic_transforms.py   # pure Anthropic↔OpenAI converters
```

## What's next

See [ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md): real
streaming, required sidecar integration coverage, replay/capture, and the
training-bridge pause/resume surface.
