# run-claude-code

Run **Claude Code** inside an Agentix sandbox with its LLM traffic **bridged**
(via `agentix.bridge`) to the host, which forwards to any OpenAI-compatible
endpoint. abridge translates Anthropic ⇄ OpenAI, so Claude Code talks to a
GPT-class model unchanged; every call is captured + traced.

```
agent (claude CLI, in sandbox) → in-sandbox abridge proxy → /abridge SIO → host
  → OpenAICompatibleClient → OpenAI-compatible endpoint
```

- `agent.py` — runs **inside** the sandbox (`agent::run_cc`): calls the plugin's
  `claude_code.run` with `ANTHROPIC_BASE_URL` (set by `bridged`).
- `main.py` — host orchestration: builds the gateway client, runs the agent via
  `bridged`, prints the captured per-call trajectory.

## Build & run

```bash
agentix build .
export OPENAI_API_KEY=sk-...
# pick the upstream model the host pins all calls to:
uv run python main.py --bundle <bundle-ref> --model gpt-4o \
    --instruction "Refactor utils.py and add tests"
```

On a restricted rootless-podman host, add `--container-engine podman --network host
--run-arg=--runtime=crun --run-arg=--cgroups=disabled` (see the `agentix-ray-build`
skill).

## Observability (LangSmith / Langfuse / any OTLP backend)

Install `agentix-trace-otel` **on the host** (it's host-side — not in the sandbox bundle). abridge emits each LLM call as a `/trace` span (OpenTelemetry GenAI conventions).
Export them by pointing `--otlp-endpoint` (+ auth headers) at any OTLP backend —
only the URL + headers differ:

```bash
# LangSmith
uv run python main.py --bundle <ref> ... \
  --otlp-endpoint https://api.smith.langchain.com/otel/v1/traces \
  --otlp-header x-api-key=$LANGSMITH_API_KEY \
  --otlp-header Langsmith-Project=run-claude-code

# Langfuse (Authorization: Basic base64(public_key:secret_key))
uv run python main.py --bundle <ref> ... \
  --otlp-endpoint https://cloud.langfuse.com/api/public/otel/v1/traces \
  --otlp-header "Authorization=Basic $(printf '%s:%s' $LANGFUSE_PUBLIC_KEY $LANGFUSE_SECRET_KEY | base64)"
```

> Docent is **not** OTLP — it ingests agent transcripts via its own SDK. To send
> there, build a Docent `AgentRun` from `bridge.store.trajectory(session_id)` (the
> captured records) and `client.add_agent_runs(...)`; that's a separate adapter.
