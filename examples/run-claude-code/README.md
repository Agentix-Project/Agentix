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
--run-arg --runtime=crun --run-arg --cgroups=disabled` (see the `agentix-ray-build`
skill).
