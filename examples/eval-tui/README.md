# eval-tui

A live [Textual](https://textual.textualize.io/) dashboard for Agentix batch
rollouts. It drives [`agentix.runner`](../../plugins/runner) and renders each
instance's progress in real time.

```text
┌─ Agentix · eval ───────────────────────────────────────────────────────────┐
│ [████████████····] 18/40 done    ✓ 11    ✗ 7    ⟳ 4 running    62.3/min     │
├──────────────────────────────────────────────┬──────────────────────────────┤
│ Instance              Status      Time  Result│ ▶ starting 40 rollouts        │
│ demo__task-000        ✓ PASS      1.2s  resolved│ ✓ PASS demo__task-000 · 1.2s │
│ demo__task-001        ✗ FAIL      1.4s  unresolv│ ✗ FAIL demo__task-001 · 1.4s │
│ demo__task-002        ⟳ scoring   …          …  │ …                            │
│ demo__task-003        ⟳ agent     …          …  │                              │
└──────────────────────────────────────────────┴──────────────────────────────┘
 q Quit   d Dark/Light
```

- **Summary bar** — progress, resolved/failed counts, in-flight, throughput.
- **Grid** — one row per instance, live status (`pending → setup → agent →
  scoring → PASS/FAIL/skip/error`), wall time, result.
- **Event log** — terminal outcomes as they land.

Phase transitions are observed by wrapping the dataset/agent adapters
(`_adapters.py`), so `agentix.runner` itself is unchanged.

## Try it (no Docker)

```bash
cd examples/eval-tui
uv sync
uv run agentix-eval-tui --demo 40 --n-concurrent 6
```

## Real runs

Point it at adapters and a provider, just like `agentix-run`:

```bash
uv run agentix-eval-tui \
    --dataset my_pkg:dataset \
    --agent my_pkg:agent \
    --provider docker \
    --bundle eval:0.1.0 \
    --model claude-3-5-sonnet-latest \
    --n-concurrent 8
```

## Test

```bash
uv sync --extra dev
uv run pytest        # headless pilot test (Textual run_test), no Docker
```
