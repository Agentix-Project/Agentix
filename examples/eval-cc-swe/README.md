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
├── default.nix        — claude CLI + git (Nix-pinned)
├── README.md
├── cc.py              — sandbox: `cc.run(...)` — spawns the proxy + runs claude
├── proxy.py           — sandbox: Anthropic↔OpenAI proxy used by cc.py
├── swe.py             — sandbox: `swe.clean`, `swe.get_patch`, `swe.eval`
└── runner.py          — host: orchestrator + abridge wiring
```

## Architecture

```
                host                                  sandbox(es) per instance
   ┌────────────────────────────┐         ┌────────────────────────────────────┐
   │ python -m runner           │         │  base: sweb.eval.x86_64.<id>:latest│
   │                            │         │  overlay: eval-cc-swe:<ver>        │
   │ # agent sandbox            │  ─────► │    swe.clean(/testbed, base)       │
   │   c.remote(swe.clean,…)    │         │    cc.run(...)                     │
   │   c.remote(cc.run,…)       │         │      ↳ proxy on 127.0.0.1:<port>   │
   │   c.remote(swe.get_patch,…)│         │      ↳ claude --print … via proxy  │
   │   c.traces() → abridge     │ ◄────── │      ↳ proxy emits llm_request/    │
   │   c.attach_logging(…)      │ ◄────── │        llm_response via trace.emit │
   │                            │         │    swe.get_patch(/testbed)         │
   │                            │         └────────────────────────────────────┘
   │ # eval sandbox (fresh)     │         ┌────────────────────────────────────┐
   │   c.remote(swe.eval,…)     │  ─────► │  same base + overlay; new container│
   └────────────────────────────┘         │    apply patch (GIT_APPLY_CMDS)    │
                                          │    bash /eval.sh                   │
                                          │    get_eval_report                 │
                                          └────────────────────────────────────┘
```

Routing is by `fn.__module__`: `cc.run` lands on a worker for module
`cc`, `swe.eval` on a worker for `swe`. The framework auto-registers
each module on first dispatch — no entry-point declaration needed.

### Trace flow

1. `runner` opens `RuntimeClient.traces(call_id=...)` and feeds the
   stream into `abridge.correlate`. Each instance's rollout file
   lands at `runs/<id>.rollouts.jsonl`.
2. `runner` also calls `client.attach_logging("agentix.sandbox")`, so
   any `logger.info(...)` inside `cc.run` / `swe.*` / `proxy.py` shows
   up in the host's logging stream — no extra plumbing on the user's
   side. Worker boot installs the bridge automatically.
3. Inside the sandbox, `cc.run` sets the trace `call_id` on the
   contextvar before launching the proxy + claude. Every
   `trace.emit(...)` from the proxy inherits that key, so abridge
   correlates an entire round of LLM round-trips into one `Rollout`.

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
- **Runtime image** (`SandboxConfig.runtime_image`): the bundle this
  directory produces. Brings in `claude`, `git`, the in-sandbox
  proxy, the `agentix-server` worker, and the `swebench` Python
  package, all under `/nix/runtime`.

## Install, build, run

```bash
# host-side: enables `from cc import run`, `from swe import …`,
# and `from runner import main` for typed dispatch
pip install -e .

# package the project + every declared dep into one image
agentix build . -o eval-cc-swe:0.1.0

# (optional) pre-pull the SWE-bench eval images you plan to score
docker pull swebench/sweb.eval.x86_64.django_1776_django-11099:latest

# Point at any OpenAI-compatible backend
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

python -m runner --limit 5
# or specific instances:
python -m runner --instance-id django__django-11099 --instance-id sympy__sympy-20212

# CI-only ground-truth check: exits on the first unresolved instance.
python -m ci_runner --num-shards 20 --shard-index 0 --out runs/ground-truth-shard-0
```

Runner flags (all optional unless noted):

- `--bundle-image`        runtime bundle to overlay (default `eval-cc-swe:0.1.0`)
- `--swebench-namespace`  docker namespace for the eval images (default `swebench`)
- `--swebench-tag`        tag on those images (default `latest`)
- `--arch`                `x86_64` or `arm64`
- `--openai-base-url`     OpenAI-compatible endpoint (env `OPENAI_BASE_URL`)
- `--openai-api-key`      bearer token for that endpoint (env `OPENAI_API_KEY`, required)
- `--openai-model`        model name the proxy forwards each request as (env `OPENAI_MODEL`)
- `--cc-timeout`          wall-clock budget for claude (default 1800s)
- `--eval-timeout`        wall-clock budget for the eval script (default 1800s)
- `--max-turns`           forwarded to the claude CLI
- `--out`                 directory for per-instance `.patch`, `.json`, `.rollouts.jsonl`

## Output

```
[django__django-11099] agent sandbox: swebench/sweb.eval.x86_64.django_1776_django-11099:latest
[django__django-11099] HEAD=d4b3eed40d92
[django__django-11099] running claude (model=gpt-4o-mini)
[django__django-11099] claude exit=0
[django__django-11099] patch_bytes=412  call_id=eval-cc-swe-django__django-11099-9af2b6cd
[django__django-11099] eval sandbox
[django__django-11099] PASS  patch_applied=True  resolved=2/2  regressions=0  (847.3s)

1/1 resolved
```

Per-instance artifacts in `runs/`:

- `runs/<id>.patch`            unified diff applied to `/testbed`
- `runs/<id>.json`             `{resolved, patch_applied, apply_cmd, fail_to_pass, pass_to_pass, ...}`
- `runs/<id>.rollouts.jsonl`   abridge Rollout per call_id (LLM turns + steps)
- `runs/summary.json`          list of every per-instance summary
