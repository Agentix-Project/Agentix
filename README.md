<div align="center">

# Agentix

**A typed Python closure runtime for Docker sandboxes.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Agentiix/Agentix)](https://github.com/Agentiix/Agentix)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

## ✨ What it is

A small framework for packaging any command as a **typed Python closure** — a Nix-built Docker image that ships a Python package the runtime imports in-process — and composing many closures into a single sandbox. Calls look like local Python (`await c.remote(claude_code.run, instruction="...")`) and execute inside the container with full IDE / mypy support.

Scope for v0.1.0 is deliberately narrow: closure packaging, sandbox composition, runtime server, typed remote dispatch. Higher-level abstractions (agent adapters, dataset runners, benchmark orchestration) are **out of scope for this release** — they'll be layered on top once the substrate settles.

## 📦 Build & package

Only Docker is required on the host. Nix runs inside a `nixos/nix` builder stage of the closure Dockerfile.

```bash
git clone https://github.com/Agentiix/Agentix.git
cd Agentix
pip install -e '.[dev]'

# Runtime + mock closures share tests/closure-docker/Dockerfile for the repo's own builds:
docker build -t agentix/runtime:0.1.0      -f tests/closure-docker/Dockerfile .
docker build -t agentix/mock-agent:0.1.0   -f tests/closure-docker/Dockerfile tests/closures/mock-agent
docker build -t agentix/mock-dataset:0.1.0 -f tests/closure-docker/Dockerfile tests/closures/mock-dataset
```

### Writing your own closure

A closure is a Python package shipped inside a Docker image:

1. Build a Python package at `agentix_closures/<name>/` with three files:
   - `__init__.py` — typed stub signatures (body: `raise NotImplementedError`)
   - `_impl.py` — actual implementation (only the sandbox runs this)
   - `_register.py` — `def register() -> Dispatcher` that binds each stub to its impl
2. Author a `default.nix` that emits the package contents under `entry/python/agentix_closures/<name>/` plus a `manifest.json` declaring `package = "agentix_closures.<name>"`.
3. Author a Dockerfile that builds the derivation and produces a final image satisfying the closure convention: `VOLUME /nix`, `/nix/store/<hash>-*`, `/nix/entry/python/...`, `/nix/entry/manifest.json`.

See `tests/closure-docker/Dockerfile`, `tests/closures/mock-agent/` and `docs/closure-protocol.md` for the full ABI.

## 🚀 Quick start

```python
import asyncio
from agentix import DockerDeployment, RuntimeClient, SandboxConfig
from agentix_closures import mock_agent  # typed stubs


async def main():
    deployment = DockerDeployment()
    config = SandboxConfig(
        image="ubuntu:24.04",
        runtime="agentix/runtime:0.1.0",
        closures=["agentix/mock-agent:0.1.0"],
    )
    async with deployment.session(config) as sandbox:
        async with RuntimeClient(sandbox.runtime_url) as c:
            print(await c.run("uname -a"))
            result = await c.remote(mock_agent.run, instruction="hello")
            print(result.patch)  # result: mock_agent.RunResult — fully typed

asyncio.run(main())
```

Under the hood the deployment:

1. For each closure image, populates a per-image named volume keyed by image digest (`docker run --rm -v vol:/nix <image> true` — Docker's own volume-init-from-image rule does the copy, idempotent).
2. Starts the sandbox with `-v agentix-closure-<digest>:/mnt/c<digest>:ro` per closure + `--tmpfs /nix`.
3. The sandbox's entrypoint builds a `/nix/store` symlink forest from each mounted closure's store contents, then execs `/mnt/runtime/entry/bin/start`.
4. The runtime server's startup scans `/mnt/*`, imports each closure's Python package, and calls `<package>._register.register()` to obtain a `Dispatcher`. Contents are fixed for the sandbox's lifetime.

## 🏗️ Architecture

```
Orchestrator ──HTTP /_remote──► Runtime Server ──in-process call──► Closure impl
```

| Component | Role |
|---|---|
| Runtime server | `/health`, `/exec`, `/upload`, `/download`, `/closures`, and the single typed-dispatch endpoint `POST /_remote`. |
| Closure | Nix-built Docker image shipping a Python package under `/nix/entry/python/agentix_closures/<name>/`. The runtime imports it. |
| Deployment | Creates sandboxes, populates per-closure named volumes, bootstraps the runtime. |

See `docs/architecture.md` and `docs/closure-protocol.md` for protocol details.

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md).

## 🤝 Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## 📄 License

[MIT License](LICENSE)
