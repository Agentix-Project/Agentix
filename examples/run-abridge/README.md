# run-abridge

Run an agent **inside** an Agentix sandbox and let **abridge** tunnel its
LLM traffic back to the host. The agent (`agent.py`) is ordinary OpenAI
SDK code — it never sees the real key and has no proxy/tracing wiring.
`main.py` (host) registers the bridge consumer, runs the agent via
`bridged`, and reads back the captured trajectory.

```
agent (sandbox)  ──http 127.0.0.1──▶  abridge proxy  ──/abridge SIO──▶  host
                                                                          │
                                              real key + forward to       ▼
                                          OpenAI / OpenRouter / vLLM / your gateway
```

Contrast with [`run-mini-swe-agent`](../run-mini-swe-agent): there the
model stays host-side and only shell commands enter the sandbox. Here the
whole agent runs in the sandbox — the case abridge exists for (e.g. an
E2B/Daytona sandbox that can't reach your private inference directly).

## Build the bundle

```bash
agentix build .
```

## Run

```bash
export OPENAI_API_KEY=sk-...
# Optional: point at a different OpenAI-compatible endpoint
# export OPENAI_BASE_URL=https://openrouter.ai/api/v1

uv run python main.py --bundle <bundle-ref> --task "Explain RL rollouts in one sentence."
```

You'll see the agent's answer, then the per-call trajectory captured on
the host (`session_id`, family, token usage). Each call is also a
`/trace` span (OTel GenAI conventions) — register a `trace.Processor` to
export it.

## How it fits together

- **Host** `OpenAICompatibleClient(base_url=..., model=..., store=...)` —
  forwards to any OpenAI-compatible endpoint and captures every call.
  `model=` pins the upstream model regardless of what the agent requested.
- **Sandbox** `sandbox.remote(bridged, solve, task, _bridge=BridgeConfig(session_id=...))` —
  `bridged` starts the proxy, sets `OPENAI_BASE_URL`/`OPENAI_API_KEY`,
  runs the agent, and tears the proxy down.
- **Readback** `store.trajectory(session_id)` — the agent-eye text
  trajectory. Token-level data (ids/logprobs) belongs to your gateway,
  keyed by the same `session_id` (abridge forwards it as `x-session-id`).
