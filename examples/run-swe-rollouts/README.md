# run-swe-rollouts

The `eval-cc-swe` flow expressed through [`agentix.runner`](../../plugins/runner):
two small adapters plus one `run_rollouts(...)` call replace the hand-written
per-instance orchestration.

- **`SweDataset`** — enumerates SWE-bench rows, builds each task image, resets
  `/testbed` to the base commit, and scores a patch with the official harness
  (`agentix.plugins.datasets.swe`).
- **`ClaudeCodeAgent`** — opens an abridge `Proxy` session on the sandbox,
  runs the `claude` CLI against the in-sandbox tunnel, and extracts the diff
  with `agentix.bash.run`. The real provider call stays on the host
  (`AnthropicFromOpenAIClient` owns the upstream call and the translation).
- **`GroundTruthAgent`** (`--ground-truth`) — submits each row's gold patch,
  reusing the identical scoring path for harness validation.

## Build

```bash
cd examples/run-swe-rollouts
uv sync
uv run agentix build . --name run-swe-rollouts:0.1.0 --platform linux/amd64 \
    --output dist/run-swe-rollouts.bundle.tar
BUNDLE=$(uv run agentix deploy docker dist/run-swe-rollouts.bundle.tar --platform linux/amd64 \
    | awk -F' -> ' '/^bundle -> /{print $2}')
```

## Run

```bash
# Agent rollouts against any OpenAI-compatible upstream:
OPENAI_BASE_URL=https://example.com/v1 OPENAI_API_KEY=sk-... UPSTREAM_MODEL=your-model \
uv run python main.py --bundle "$BUNDLE" --limit 5 --concurrency 4

# Ground-truth harness check (no agent, no key needed):
uv run python main.py --bundle "$BUNDLE" --ground-truth --fail-on-unresolved

# CI-style sharding — run one slice of the split (nightly-CI runs 20 of these):
uv run python main.py --bundle "$BUNDLE" --ground-truth --num-shards 20 --shard-index 0
```

Sharding is round-robin and deterministic; `--limit` applies per shard, and a
shard with no rows (more shards than rows) exits 0. `--instance-id` selects
rows explicitly and cannot be combined with `--num-shards`/`--shard-index`.

Per-instance summaries land in `runs/<instance_id>.json`, plus a combined
`runs/summary.json`.
