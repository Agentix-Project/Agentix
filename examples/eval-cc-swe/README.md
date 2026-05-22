# eval-cc-swe — evaluate Claude Code on SWE-bench Verified

End-to-end example: per instance, spin up a sandbox on top of the
official SWE-bench eval image, run Claude Code against `/testbed` via
an in-sandbox Anthropic→OpenAI proxy (so any OpenAI-compatible model
serves), extract the diff, then spin up a fresh sandbox to apply the
diff and score it with the official harness. LLM round-trips surface
on the host through `agentix.trace` and feed abridge for rollout logs.

```
examples/eval-cc-swe/
├── pyproject.toml     — project + deps
├── README.md
└── runner.py          — host orchestrator + sandbox `get_patch(...)`

plugins/agents/claude-code/
├── src/
│   └── __init__.py    — sandbox: `claude_code.run(...)`
└── default.nix        — Claude Code CLI

plugins/datasets/swebench/
├── src/
│   ├── __init__.py    — public exports
│   ├── env.py         — sandbox env preparation
│   └── score.py       — sandbox scoring
└── default.nix        — git/patch/libstdc++ for SWE-bench images
```

## Architecture

```
                host                                  sandbox(es) per instance
   ┌────────────────────────────┐         ┌────────────────────────────────────┐
   │ python -m runner           │         │  base: sweb.eval.x86_64.<id>:latest│
   │                            │         │  overlay: eval-cc-swe:<ver>        │
   │ # client1 / agent sandbox  │  ─────► │    agentix.plugins.datasets.swe.prepare_env│
   │   c.remote(swe.prepare_env)│         │    agentix.agents.claude_code.run  │
   │   c.remote(cc.run,…)       │         │      ↳ claude --print via abridge  │
   │   c.remote(get_patch,…)    │         │    runner.get_patch               │
   │   c.traces() → abridge     │ ◄────── │                                    │
   │   c.attach_logging(…)      │ ◄────── │                                    │
   │                            │         └────────────────────────────────────┘
   │ # client2 / score sandbox  │         ┌────────────────────────────────────┐
   │   c.remote(swe.score,…)    │  ─────► │  same base + overlay; new container│
   │                            │         │    reset to base_commit            │
   └────────────────────────────┘         │    apply patch (GIT_APPLY_CMDS)    │
                                          │    apply SWE test patch            │
                                          │    run targeted tests + score      │
                                          └────────────────────────────────────┘
```

Use it as `from agentix.plugins.datasets import swe` or
`from agentix.plugins.datasets.swe import prepare_env, score`.
`score(...)` prepares `/testbed` itself before applying the submitted
patch.

### Trace flow

1. `runner` opens `RuntimeClient.traces(call_id=...)` and feeds the
   stream into `abridge.correlate`. Each instance's rollout file
   lands at `runs/<id>.rollouts.jsonl`.
2. `runner` also calls `client.attach_logging("agentix.sandbox")`, so
   any `logger.info(...)` inside `claude_code.run` / `swe.*` shows
   up in the host's logging stream — no extra plumbing on the user's
   side. Worker boot installs the bridge automatically.
3. Inside the sandbox, `claude_code.run` points Claude Code at the
   Anthropic-compatible abridge service, so abridge correlates LLM
   round-trips into one `Rollout`.

The trace backlog is replayed from cursor 0 on each `client.traces()`
subscription — there is no buffering / drop tradeoff.

### Why two sandboxes per instance

The eval container has to start from a known clean state at
`base_commit` with the `test_patch` not yet applied — exactly what
`swebench.harness.run_evaluation.run_instance` expects. Tearing down
the agent container and bringing up a fresh one for the eval pass is
both the simplest way to guarantee that and a faithful match to the
official harness flow.

## Image choices

- **Base image** (`SandboxConfig.image`):
  `swebench/sweb.eval.x86_64.<instance_id>:latest`. Pre-built by the
  SWE-bench project — already contains the `testbed` conda env and
  `/testbed` cloned at `base_commit`. Runner constructs the tag from
  the instance row (the lowercase id with `__` rewritten to `_1776_`).
- **Bundle image** (`SandboxConfig.bundle`): the bundle this
  directory produces. Brings in `claude`, `git`, `agentix-server`, the
  Claude Code agent plugin, and the SWE dataset plugin, all under
  `/nix/runtime`.

## Install, build, run

```bash
# host-side: installs the example plus local agent/dataset plugins
uv sync

# package the project + every declared dep into one image
uv run agentix build . --name eval-cc-swe:0.2.0

# (optional) pre-pull the SWE-bench eval images you plan to score
docker pull swebench/sweb.eval.x86_64.django_1776_django-11099:latest

# Point at any OpenAI-compatible backend
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export UPSTREAM_MODEL=gpt-4o-mini

python -m runner --limit 5
# or specific instances:
python -m runner --instance-id django__django-11099 --instance-id sympy__sympy-20212

# Ground-truth check: skips client1 and scores dataset patches directly.
uv run python -m runner --ground-truth --fail-on-unresolved \
  --concurrency 20 --out runs/ground-truth
```

Runner flags (all optional unless noted):

- `--bundle`        runtime bundle to overlay (default `eval-cc-swe:0.2.0`)
- `--swebench-namespace`  docker namespace for the eval images (default `swebench`)
- `--swebench-tag`        tag on those images (default `latest`)
- `--arch`                `x86_64` or `arm64`
- `--openai-base-url`     OpenAI-compatible endpoint (env `OPENAI_BASE_URL`)
- `--openai-api-key`      bearer token for that endpoint (env `OPENAI_API_KEY`, required)
- `--upstream-model`      model name the proxy forwards each request as (env `UPSTREAM_MODEL`)
- `--response-model`      model name echoed to Claude Code (env `RESPONSE_MODEL`)
- `--cc-timeout`          wall-clock budget for claude (default 1800s)
- `--eval-timeout`        wall-clock budget for SWE test scoring (default 1800s)
- `--max-turns`           forwarded to the claude CLI
- `--concurrency`         max instances to run at once (default 1)
- `--out`                 directory for per-instance `.patch`, `.json`, `.rollouts.jsonl`

## Output

```
[django__django-11099] agent sandbox: swebench/sweb.eval.x86_64.django_1776_django-11099:latest
[django__django-11099] HEAD=d4b3eed40d92
[django__django-11099] running claude (model=gpt-4o-mini)
[django__django-11099] claude exit=0
[django__django-11099] patch_bytes=412  call_id=eval-cc-swe-django__django-11099-9af2b6cd
[django__django-11099] score sandbox
[django__django-11099] PASS  patch_applied=True  resolved=2/2  regressions=0  (847.3s)

1/1 resolved
```

Per-instance artifacts in `runs/`:

- `runs/<id>.patch`            unified diff applied to `/testbed`
- `runs/<id>.json`             `{resolved, patch_applied, timed_out, test_status, ...}`
- `runs/<id>.rollouts.jsonl`   abridge Rollout per call_id (LLM turns + steps)
- `runs/summary.json`          list of every per-instance summary
