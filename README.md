<div align="center">

# Agentix

**The bridge between agents, evaluation, RL training, and LLM serving.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Agentiix/Agentix)](https://github.com/Agentiix/Agentix)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-agentiix.github.io-blue)](https://agentiix.github.io/)

[Documentation](https://agentiix.github.io/) | [Supported Integrations](#supported-integrations) | [Cookbook](https://github.com/Agentiix/agentix-cookbook) | [RL Bridge](https://github.com/Agentiix/abridge)

</div>

## Overview

**Agentix** is the **execution, tracing, and integration bridge**
between agents and your LLM serving, RL post-training, and evaluation
infrastructure. It gives trainers, evaluators, and agent builders one
typed Python interface for running agents in isolated rollout
containers, capturing LLM/tool traces, and routing those traces to
benchmark scorers, RL buffers, observability sinks, or custom serving
providers.

Use it when you need to connect Claude Code, Codex, Aider,
mini-swe-agent, OpenHands, or an in-house agent to SWE-bench,
custom evals, an LLM proxy, or an RL data buffer without writing a
bespoke runner for every agent x benchmark x training stack.

Each agent, dataset, or tool is a regular Python package. Call it from
your trainer or evaluator with typed remote dispatch:
`c.remote(fn, ...)` reads `fn`'s signature, so Pyright can infer return
types across the host-to-container boundary.

## Why Agentix

- **One bridge, many stacks.** Shell commands, agent CLIs, Python
  frameworks, dataset scorers, and file operations all use the same
  `RuntimeClient.remote(...)` call path.
- **Isolation without glue sprawl.** Every integration runs in its own
  dependency environment inside the same rollout container, so
  incompatible agent stacks can be bundled together.
- **Training-ready trace flow.** LLM calls and tool activity can stream to
  [abridge](https://github.com/Agentiix/abridge) (the official
  Agentix extension for RL training), observability sinks, or your own
  collector.
- **Benchmarks stay composable.** Agent execution and scoring remain
  separate namespaces, which makes it easy to swap agents, scorers,
  and deployment backends independently.

## What Agentix Bridges

| From | Agentix layer | To |
|---|---|---|
| Agent CLIs and Python frameworks | Isolated namespace workers with typed remote dispatch | Evaluators, trainers, and orchestration code |
| Tool calls and LLM traffic | Structured trace capture and fan-out | Observability, replay, reward, and dataset pipelines |
| Rollout traces | [abridge](https://github.com/Agentiix/abridge) correlation and sink protocol | RL buffers such as slime / verl, or custom serving stacks |
| Local and hosted sandboxes | `agentix.deployment` backends | Docker today; Daytona, E2B, and third-party backends as plugins |

## Key Features

- **Run any agent in an isolated rollout container.** Bring a CLI
  binary, a Python framework, or your own package. Built-in recipe:
  [Claude Code](https://github.com/Agentiix/agentix-cookbook/tree/main/claude-code).
- **Score against any benchmark.** Built-in:
  [SWE-bench Verified](https://github.com/Agentiix/agentix-cookbook/tree/main/swebench),
  wrapping the official
  [`swebench`](https://github.com/swe-bench/SWE-bench) harness's test
  specs, log parsers, and grading.
- **Bridge to RL training and serving.** Every LLM call and tool
  invocation streams out as a structured trace.
  [abridge](https://github.com/Agentiix/abridge) — Agentix's host-side
  trace bridge — consumes that stream, correlates events by rollout,
  and hands them to a framework-specific sink (RL trainer data buffer,
  serving / evaluation pipeline, or your own).
- **Pluggable execution backends.** `local` (Docker), `daytona`, and
  `e2b` built in; Fly, Modal, Kubernetes via
  `pip install agentix-deployment-<name>`.
- **Typed dispatch across the bridge.** Container methods autocomplete
  like local functions; your editor knows the kwargs and return types.
  Three call shapes (unary / streaming / bidirectional) are
  auto-detected from your function signature.
- **Trace fan-out and observability.** Every integration's
  `trace.emit(...)` events can ship to OpenTelemetry, Sentry, or your
  own bus with one `agentix.trace.subscribe(fn)` call.

## Supported Integrations

### Agents

- **Claude Code** — [recipe](https://github.com/Agentiix/agentix-cookbook/tree/main/claude-code)
- CLI binaries (Codex, Aider), Python frameworks
  ([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
  [swe-agent](https://github.com/SWE-agent/SWE-agent),
  [OpenHands](https://github.com/All-Hands-AI/OpenHands)),
  or your own — wrap with the
  [agent integration guide](https://agentiix.github.io/integrate-agent).

### Benchmarks

- **SWE-bench Verified** — [recipe](https://github.com/Agentiix/agentix-cookbook/tree/main/swebench),
  built on the official
  [`swebench`](https://github.com/swe-bench/SWE-bench) package's
  `make_test_spec` + `get_eval_report`

### Sandbox Primitives

- **bash** — shell execution inside the rollout container. Ships with
  [`agentix-runtime-basic`](https://github.com/Agentiix/Agentix-Runtime-Basic).
- **files** — upload, download, and edit files in the rollout container.
  Same wheel.

### Execution Backends

- `local` — Docker-based; ships with
  [`agentix-deployment-docker`](https://github.com/Agentiix/Agentix-Deployment-Docker).
- `daytona` — [`agentix-deployment-daytona`](https://github.com/Agentiix/Agentix-Deployment-Daytona).
- `e2b` — [`agentix-deployment-e2b`](https://github.com/Agentiix/Agentix-Deployment-E2B).
- Third-party — `pip install agentix-deployment-<name>`.

### RL Frameworks / Serving Providers

- **abridge** — Agentix's host-side rollout bridge; correlates runtime
  trace events into per-rollout records and pushes them to sinks.
- **slime / verl** — RL post-training adapters can consume abridge
  rollouts and append them to framework-specific buffers.
- **Custom serving or evaluation stacks** — implement `abridge.Sink`
  or subscribe directly with `agentix.trace.subscribe(fn)`.

## Architecture

```
Orchestrator ──HTTP /_remote──► Runtime Server ──fork──► Namespace worker (per integration)
   (trainer)                       (multiplexer)            (own venv, own PATH)
                                        ▲
            Socket.IO /socket.io/ ◄──────┴──── streams, bidi, logs, traces
```

- **Runtime server**: one process per rollout container. Routes
  `POST /_remote` (unary) and Socket.IO events (streams / bidi / logs
  / traces) to per-integration workers spawned lazily on first
  dispatch.
- **Namespace worker**: subprocess that imports the integration using
  its own venv. Each integration's dependencies stay isolated from
  every other's — mix Aider 0.50 and OpenHands 0.20 in one container
  without resolving deps across them.
- **Deployment**: host-side backend (`local`, `daytona`, `e2b`, or a
  third-party plugin) that creates the rollout container and returns
  its `runtime_url`.

Discovery is lazy — a broken integration fails its own calls but
never blocks boot.

## Install

```bash
pip install agentix \
            agentix-runtime-basic \
            agentix-deployment-docker
```

Cookbook integrations:

```bash
git clone https://github.com/Agentiix/agentix-cookbook
pip install ./agentix-cookbook/claude-code ./agentix-cookbook/swebench
```

Framework development:

```bash
git clone https://github.com/Agentiix/Agentix && cd Agentix
pip install -e '.[dev]'
# Pair with sibling repos checked out next to Agentix/ for a working
# rollout end-to-end:
pip install -e ../Agentix-Runtime-Basic -e ../Agentix-Deployment-Docker
```

## CLI

```bash
agentix build                                                # build current project
agentix build path/to/project -o my-agent:0.1.0              # explicit path + tag
agentix deploy local --image my-agent:0.1.0                  # run a rollout container
agentix check                                                # smoke-test every installed integration
```

Multi-plugin bundles are expressed by declaring the plugins as deps
in your project's `pyproject.toml`; pip pulls them in transitively.

## Write an integration

```python
# src/agentix/myagent/__init__.py
async def run(instruction: str) -> str:
    return f"did: {instruction}"
```

```toml
# pyproject.toml
[project]
name = "agentix-myagent"
version = "0.1.0"

[project.entry-points."agentix.namespace"]
myagent = "agentix.myagent"

[tool.hatch.build.targets.wheel]
packages = ["src/agentix"]
```

After `pip install agentix-myagent`:

```python
from agentix import myagent
result = await c.remote(myagent.run, instruction="...")
```

## Two extension axes

Only things that cross the host↔container boundary need framework-level
discovery:

| Axis | Entry-point group | What it ships | Built-ins |
|---|---|---|---|
| Namespaces | `agentix.namespace` | code that runs **inside the rollout container** | (third-party only) |
| Deployments | `agentix.deployment` | backend that **provisions** the container | `local`, `daytona`, `e2b` |

Host-side hooks (trace pub/sub, spec resolvers, CLI verbs) are plain
Python — `agentix.trace.subscribe(fn)` is the single line that ships
every integration's `trace.emit(...)` events into OpenTelemetry,
Sentry, or your own bus.

## Links

- **Docs site**: [agentiix.github.io](https://agentiix.github.io/)
- **Cookbook**: [github.com/Agentiix/agentix-cookbook](https://github.com/Agentiix/agentix-cookbook)
- **RL bridge (abridge)**: [github.com/Agentiix/abridge](https://github.com/Agentiix/abridge)
- **Roadmap**: [ROADMAP.md](ROADMAP.md)
- **Contributing**: [docs/development.mdx](docs/development.mdx); conventions in [CLAUDE.md](CLAUDE.md)

## License

[MIT](LICENSE)
