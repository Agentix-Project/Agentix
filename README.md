<div align="center">

# Agentix

**Sandboxed rollouts you call like typed Python.**

Turn agents, tools, and scorers into Python callables. Package their
dependencies into runtime images. Call them from evaluators, trainers,
and orchestration code without writing a new runner for every pairing.

[![GitHub Stars](https://img.shields.io/github/stars/Agentiix/Agentix)](https://github.com/Agentiix/Agentix)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-agentiix.github.io-blue)](https://agentiix.github.io/)

[Documentation](https://agentiix.github.io/) | [Quickstart](https://agentiix.github.io/quickstart) | [Cookbook](https://github.com/Agentiix/agentix-cookbook) | [Architecture](https://agentiix.github.io/reference/architecture)

</div>

## The 10-Second Model

Agentix has two primitives:

- **Remote calls**: `client.remote(fn, *args, **kwargs)` runs a Python
  callable inside a sandbox worker. The callable is serialized with
  stdlib pickle, Python's native callable reference mechanism.
- **Bundles**: `agentix build [path]` packages a Python project and its
  declared dependencies into a deploy-ready runtime image.

```python
from agentix import RuntimeClient, SandboxConfig, session
from agentix.bash import run
from agentix.deployment.docker import DockerDeployment

config = SandboxConfig(
    image="python:3.13-slim",
    runtime_image="hello-agentix:0.1.0",
)
async with session(DockerDeployment(), config) as sandbox:
    async with RuntimeClient(sandbox.runtime_url) as client:
        result = await client.remote(run, command="echo hello from $(uname -a)")
```

The unit of composition is not a bespoke benchmark runner or agent
adapter. It is a Python callable.

## Why Agentix Exists

Agent experiments sprawl quickly. One agent needs a CLI wrapper. Another
needs a Python harness. A benchmark needs repo setup, grading scripts,
and logs. A training loop needs the same pieces batched across many
sandboxes.

Agentix collapses that matrix into one execution contract: if Python can
serialize the callable and the sandbox has its dependencies, the host can
call it.

| You have | You expose | You call |
| --- | --- | --- |
| Claude Code, Codex, Aider, OpenHands, or an internal agent | `async def run(...) -> RunResult` | `await client.remote(run, ...)` |
| Shell, files, repo setup, or local tools | `async def run(command: str) -> BashResult` | `await client.remote(bash_run, ...)` |
| SWE-bench, MLE-Bench, or an internal evaluator | `async def score(...) -> Score` | `await client.remote(score, ...)` |

## What Ships

- **Unary remote calls** (`await client.remote(fn, *args, **kwargs)`)
  across the host-to-sandbox boundary. One shape. Pickled callable +
  pickled args/kwargs go in; the pickled return value comes back.
- **One runtime worker process today** behind an internal worker backend
  boundary, so future pools or per-call isolation can stay API-compatible.
- **Tracing** as an orthogonal layer: `agentix.trace.span(...)` inside
  the sandbox produces Trace/Span/SpanEvent lifecycle events that
  cross to the host via a side-channel and dispatch to any
  `agentix.trace.Processor` the host has registered.
- **Bundle builds** from normal Python projects and `pyproject.toml`
  dependencies.
- **Optional Nix system dependencies** when a project includes
  `default.nix`.
- **Deployment backend plugins** through the `agentix.deployment` entry
  point group.

## Quickstart

Run the smallest demo from
[`agentix-cookbook/examples/hello-agentix`](https://github.com/Agentiix/agentix-cookbook/tree/main/examples/hello-agentix):

```bash
cd examples/hello-agentix
uv sync
uv run agentix build . --name hello-agentix  # builds hello-agentix:0.1.0
uv run python run.py
```

When the sandbox runs on a different CPU architecture than your build
host, build for the sandbox platform explicitly, for example
`uv run agentix build . --name hello-agentix --platform linux/amd64`.

The demo builds `hello-agentix:0.1.0`, overlays it onto
`python:3.13-slim`, then calls `agentix.bash.run` inside the sandbox:

```python
import asyncio

from agentix import RuntimeClient, SandboxConfig, session
from agentix.bash import run
from agentix.deployment.docker import DockerDeployment


async def main() -> None:
    deployment = DockerDeployment()
    config = SandboxConfig(
        image="python:3.13-slim",
        runtime_image="hello-agentix:0.1.0",
    )
    async with session(deployment, config) as sandbox:
        print(f"sandbox up at {sandbox.runtime_url}")
        async with RuntimeClient(sandbox.runtime_url) as client:
            result = await client.remote(run, command="echo hello from $(uname -a)")
            print(f"exit={result.exit_code} stdout={result.stdout!r}")


asyncio.run(main())
```

Read the full [quickstart](https://agentiix.github.io/quickstart) for the
project layout, lockfile, and runtime-image details.

## Architecture

```text
Host process
  RuntimeClient.remote(fn, *args, **kwargs)
    builds RemoteCallable from pickle.dumps(fn)
    pickles (args, kwargs) as one blob
        |
        v  Socket.IO "unary"
Sandbox
  agentix-server
        |  msgpack frame
        v
  worker subprocess
    RemoteCallable.resolve() -> fn
    pickle.loads(arguments) -> args, kwargs
    calls fn(*args, **kwargs)
    pickles the result back
```

The wire carries a single shape: unary. Cancellation has its own
event (`cancel`). Trace lifecycle events cross as a separate
broadcast channel (`trace:event`). HTTP is kept only for `/health`.
Errors stay in-band.

## Repository Map

- [`Agentix-Runtime-Basic`](https://github.com/Agentiix/Agentix-Runtime-Basic):
  sandbox primitives such as `bash` and file operations.
- [`Agentix-Deployment-Docker`](https://github.com/Agentiix/Agentix-Deployment-Docker):
  local Docker deployment backend.
- [`Agentix-Deployment-Daytona`](https://github.com/Agentiix/Agentix-Deployment-Daytona)
  and [`Agentix-Deployment-E2B`](https://github.com/Agentiix/Agentix-Deployment-E2B):
  hosted sandbox backend packages.
- [`agentix-cookbook`](https://github.com/Agentiix/agentix-cookbook):
  working integration recipes for agents and benchmarks.
- [`abridge`](https://github.com/Agentiix/abridge): rollout-to-RL-buffer
  bridge.

## Development

```bash
git clone https://github.com/Agentiix/Agentix
cd Agentix
pip install -e '.[dev]'
pytest
ruff check agentix/ tests/
```

Pair this repo with sibling backend/runtime repos checked out next to it
when testing full sandbox rollouts.

## Links

- [Docs](https://agentiix.github.io/)
- [Quickstart](https://agentiix.github.io/quickstart)
- [Remote calls](https://agentiix.github.io/concepts/remote-calls)
- [Bundles](https://agentiix.github.io/concepts/bundles)
- [Architecture](https://agentiix.github.io/reference/architecture)
- [Roadmap](ROADMAP.md)
