# agentix-runner

Library-first batch rollout runner for Agentix. Run an agent over a dataset
of instances — each in its own sandbox — and collect typed `Rollout`
records. It is built on the stable Agentix surface (`provider.session(...)`
+ `sandbox.remote(fn, ...)`) and contains no benchmark- or agent-specific
logic; datasets and agents plug in through two small Protocols.

```python
from agentix.runner import run_rollouts

rollouts = await run_rollouts(
    dataset=my_dataset,    # implements agentix.runner.Dataset
    agent=my_agent,        # implements agentix.runner.Agent
    provider=provider,     # any Agentix SandboxProvider
    bundle="eval:0.1.0",   # produced by `agentix build`
    model="claude-3-5-sonnet-latest",
    n_concurrent=8,
)
resolved = sum(r.resolved for r in rollouts)
```

An RL or eval loop calls `run_rollouts(...)` directly. The `agentix-run`
CLI is a thin wrapper for manual runs:

```bash
agentix-run --dataset my_pkg:dataset --agent my_pkg:agent \
    --provider docker --bundle eval:0.1.0 --n-concurrent 8 --out runs/
```

## Adapters

- **`Dataset`** — `instances()`, `image(inst)`, `setup(sandbox, inst) -> bool`,
  `score(sandbox, inst, patch) -> dict`.
- **`Agent`** — `solve(sandbox, inst, *, model) -> AgentResult`.

`setup`/`score` and `solve` receive the live sandbox, so they drive work with
`sandbox.remote(fn, ...)`. Each phase (agent, then scoring) runs in a fresh
sandbox so scoring always starts from a clean task image.
