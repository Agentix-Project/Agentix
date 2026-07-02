# agentix-provider-uv

A lightweight Agentix `SandboxProvider` that runs the runtime from a
**uv-materialized virtualenv** — no Docker image, no Nix bundle.

`uv` builds a venv for the target project (so its importable callables +
`agentixx` core are present), then the runtime server is launched as a local
subprocess (`python -m uvicorn agentix.runtime.server.app:app`). The worker the
server spawns inherits that interpreter, so `await sandbox.remote(fn, ...)` runs
against the project's real dependencies.

```python
from agentix.provider.base import SandboxConfig
from agentix.provider.uv import UvProvider, UvProviderConfig

# materialize from a project (must depend on agentixx)
provider = UvProvider(UvProviderConfig(project="."))
# ...or reuse a prebuilt env and skip materialization
provider = UvProvider(UvProviderConfig(reuse_venv="/path/to/venv"))

try:
    async with provider.session(SandboxConfig(image="uv", bundle="uv")) as sandbox:
        result = await sandbox.remote(my_rollout, task=task)
finally:
    await provider.aclose()   # removes a venv the provider materialized
```

`SandboxConfig.image` / `bundle` are unused (placeholders); only `env` is
honored. This backend runs on the host with **no container isolation** — use it
for fast local dev / eval / CI, and a container provider (`docker` /
`apptainer`) or managed backend for untrusted code or hard resource limits.

`providers().get("uv")` resolves after `uv sync`. There is no `agentix deploy
uv` — the runtime is materialized from source, so there is no bundle artifact.
