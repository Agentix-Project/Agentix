# eval-mini-swe-swebench

End-to-end SWE-bench (Verified / Lite) evaluation harness for the
`mini-swe-agent` integration. Mirrors the layout of
`examples/eval-cc-swe/` but swaps Claude Code for mini-swe-agent.

Per instance, two sandboxes back-to-back:

1. **Agent sandbox** — SWE-bench's per-instance eval image
   (`swebench/sweb.eval.x86_64.<id>:latest`).
   * `c.remote(agentix.plugins.datasets.swe.prepare_env)` resets
     `/testbed` to the base commit.
   * A `DefaultAgent` (mini-swe-agent + LiteLLM) is constructed on
     the host (where the API key lives) and passed into the sandbox
     via `c.remote(agentix.agents.mini_swe_agent.run, agent=...)`.
   * `c.remote(get_patch)` extracts the final unified diff.

2. **Score sandbox** — fresh `ubuntu:24.04` (no LLM access required).
   * `c.remote(agentix.plugins.datasets.swe.score, instance=...,
     patch=...)` applies the patch, runs the SWE-bench harness, and
     returns `resolved` + per-test results.

## Run (local, 1 instance smoke test)

```bash
uv sync
uv run agentix build . --format oci-image
export OPENAI_API_KEY="sk-..."
uv run python runner.py --limit 1 --deployment local
```

The default model is `openai/gpt-4o-mini`. Set `--model` to anything
LiteLLM understands (`anthropic/...`, `openrouter/...`, etc.).

## Run (full SWE-bench Verified)

```bash
uv run python runner.py \\
    --split test \\
    --limit 500 \\
    --concurrency 8 \\
    --deployment local \\
    --model openai/gpt-4o-mini
```

Expect ~hours and $50–$200 in API spend with `gpt-4o-mini`,
proportionally more for frontier models. Results land in
`eval-results/<instance_id>.json` plus a `summary.json` with the
aggregate `resolved_rate`.

## Run on an HPC host (apptainer)

Build a portable tar bundle and run with the apptainer backend:

```bash
uv run agentix build . --format tar --output /path/to/eval-mini-swe-swebench.tar
uv run python runner.py \
    --deployment apptainer \
    --bundle /path/to/eval-mini-swe-swebench.tar \
    --split test --limit 50 --concurrency 8
```

`ApptainerDeployment` is parameterised over `AGENTIX_APPTAINER_BIN`
and `AGENTIX_APPTAINER_CACHE`, so a non-default install path just
needs the env vars set on the host.

## CLI

```
--dataset           default princeton-nlp/SWE-bench_Verified
--split             default test
--deployment        local | apptainer (default: local)
--bundle            local: docker image tag; apptainer: tar bundle path
--instance-namespace default `swebench` (matches docker hub)
--instance-tag      default `latest`
--instance-arch     default `x86_64` (override on arm hosts)
--score-image       default `ubuntu:24.04`
--platform          optional `--platform` for both task and bundle (linux/amd64, ...)
--limit             default 1 (smoke test)
--concurrency       default 1
--model             default `openai/gpt-4o-mini` (any LiteLLM model spec)
--cost-limit        per-instance cost ceiling (USD); 0 = no limit
--agent-timeout     per-instance wall-clock budget (seconds)
--output            results directory (default `eval-results/`)
--instances         explicit instance_id allow-list
```

## Outputs

* `eval-results/<instance_id>.json` — one file per instance, with
  the agent phase, score phase, durations, and any error trace.
* `eval-results/summary.json` — `{total, agent_ok, scored_ok,
  resolved, resolved_rate}`.

Each instance file is self-contained — no implicit ordering or
state — so re-running missing instances is trivial.
