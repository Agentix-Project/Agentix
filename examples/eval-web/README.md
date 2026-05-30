# agentix-eval-web

A live **web dashboard** for Agentix batch rollouts — a FastAPI server and a
single dark page that streams a run as it happens over a WebSocket.

It's the web counterpart of [`agentix-eval-tui`](../eval-tui): both are thin
frontends over **one engine**. This app reuses the TUI's demo provider, its
phase-tracing adapters, and the catalog introspection, and drives
`agentix.runner.run_rollouts` exactly as the TUI does — so a rollout looks the
same whether you watch it in the terminal or the browser.

## Run

```bash
uv run --project examples/eval-web agentix-eval-web      # → http://127.0.0.1:8000
# options: --host 0.0.0.0 --port 8080 --reload
```

Open the page and hit **▶ run**. No Docker needed — the demo provider runs
in-process with seeded, deterministic outcomes.

## What you see

- **Instance grid** — one cell per rollout, recoloured live through its phases
  (`setup → agent → scoring`) to its verdict (resolved / failed).
- **Summary** — total / done / resolved / failed / running + throughput.
- **Throughput sparkline** — completions over time.
- **Event log** — per-instance verdicts as they land.
- **Catalog** — the installed `agentix*` packages and entry points.

## API

| Route | What |
|-------|------|
| `GET /` | the dashboard |
| `GET /api/catalog` | installed Agentix ecosystem (JSON) |
| `WS /ws/run?n=&concurrency=` | start a demo run; streams `start` / `phase` / `result` / `done` events |

## Aesthetic

Dark, monospace, blackletter wordmark — a nod to the
[psyche.network](https://psyche.network/runs) run-board look. Tune the palette
in the `:root` CSS variables of `eval_web/static/index.html`.

Standalone project (own lock); path-depends on the monorepo + the TUI example.
Not type-/test-checked by core CI.
