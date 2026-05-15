<div align="center">

# Agentix

**Typed Python namespaces for sandbox-based agent workflows.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Agentiix/Agentix)](https://github.com/Agentiix/Agentix)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

</div>

## What it is

A small framework that lets you compose agent / dataset / primitive code into a sandbox and call it from your trainer or harness as if it were local typed Python:

```python
from agentix import RuntimeClient
from agentix.bash import Bash
from agentix.claude_code import ClaudeCode      # pip install agentix-claude-code
from agentix.swebench import SWEBench           # pip install agentix-swebench

async with RuntimeClient(sandbox_url) as c:
    task   = await c.remote(SWEBench.get_task, idx=42)
    patch  = await c.remote(ClaudeCode.run, instruction=task.problem)
    reward = await c.remote(SWEBench.score, idx=42, patch=patch)
```

Every extension is a normal pip-installable distribution. There is no custom config file, no decorator at import time, no per-framework registry call: the user installs a wheel and the framework discovers it via Python entry points.

Six extension axes — closures used to be the only one; this version adds five more on the same mechanism (see "Extending Agentix" below).

## Install

```bash
pip install agentix
# Plus whichever namespaces you actually need:
pip install agentix-bash agentix-files
pip install agentix-claude-code agentix-swebench   # examples — not yet on PyPI
```

For local development of the framework itself:

```bash
git clone https://github.com/Agentiix/Agentix.git
cd Agentix
pip install -e '.[dev]'
pip install -e primitives/bash -e primitives/files   # the bundled primitives
```

## CLI

```bash
agentix plugins                                # list every installed plugin across every axis
agentix build  primitives/bash                 # build a single namespace image
agentix install bash files claude-code -o my-agent:0.1.0   # bundle several namespaces
agentix deploy local --image my-agent:0.1.0    # run a sandbox + connect
agentix check                                  # smoke-import every installed namespace
```

Every subcommand is itself an entry-point plugin under the `agentix.cli` group — `pip install your-extension` plus one TOML block makes `agentix <new-command>` work without patching the framework.

## Writing a namespace

A namespace is a class whose `@staticmethod` methods are the remote-callable surface. The class is a pure namespace — methods carry no `self`, no instance state.

```python
# src/agentix/myagent/__init__.py
from agentix.namespace import Namespace

class MyAgent(Namespace):
    @staticmethod
    async def run(instruction: str) -> str:
        ...
```

Ship it with one entry-point declaration:

```toml
# pyproject.toml
[project]
name = "agentix-myagent"
version = "0.1.0"

[project.entry-points."agentix.namespace"]
myagent = "agentix.myagent:MyAgent"

[tool.hatch.build.targets.wheel]
packages = ["src/agentix"]
```

`pip install agentix-myagent` is the entire setup. Caller-side:

```python
from agentix.myagent import MyAgent
result = await c.remote(MyAgent.run, instruction="...")
```

The framework's `agentix/__init__.py` extends `__path__` so `agentix.<your-namespace>` resolves natively; PEP 420 namespace packages mean multiple dists can install peer entries under `agentix/` without colliding. Reserved framework subpackages (`agentix.cli`, `agentix.dispatch`, `agentix.deployment`, …) are listed in [CLAUDE.md](CLAUDE.md).

## Extending Agentix

Six axes, all entry-point discovered with the same mechanism:

| Axis | Entry-point group | Semantics | Built-ins |
|---|---|---|---|
| Namespaces | `agentix.namespace` | typed remote-callable surface | (third-party only) |
| Deployments | `agentix.deployment` | sandbox lifecycle, select-one by name | `local` / `daytona` / `e2b` |
| Trace sinks | `agentix.trace_sink` | fan-out trace event consumers | (third-party only) |
| Spec resolvers | `agentix.spec_resolver` | CLI input → namespace spec, chain | `path` / `image` / `local_repo` / `pypi` |
| Wire patterns | `agentix.wire_pattern` | call-shape extensions | `unary` / `stream` / `bidi` |
| CLI subcommands | `agentix.cli` | `agentix <name>` discovery | `build` / `install` / `deploy` / `check` / `plugins` |

Every axis looks the same to a plugin author:

```toml
[project.entry-points."agentix.<axis>"]
my-thing = "module:Thing"
```

> The quotes around the group name are TOML syntax — `agentix.deployment` contains a dot, and TOML treats dots in `[a.b.c]` as table-key separators. Quoting forces it to be a single key. Every framework with a dotted group name does this (`flask.commands`, `mkdocs.plugins`, `sphinx.builders`, …).

See [`docs/plugins.md`](docs/plugins.md) for the full plugin authors guide — one section per axis with working examples.

## Architecture

```
Orchestrator ──HTTP /_remote──► Runtime Server ──in-process call──► Namespace impl
                  (or)                            (Dispatcher)
            Socket.IO /socket.io/  ◄─── streams, bidi, logs, traces ───►
```

| Component | Role |
|---|---|
| Runtime server | `/health`, `/namespaces`, `/_remote` (unary), `/socket.io/` (streams/bidi/logs/traces), `/_llm/<provider>/<path>` (LLM-proxy fan-in) |
| Namespace | Python class registered under `agentix.namespace` entry point; methods called via `c.remote(...)` |
| Deployment | Sandbox CRUD plugin under `agentix.deployment`; `local` (Docker) is built in |
| WirePattern | Pluggable call-shape strategy — built-ins are unary / stream / bidi |
| Trace sink | Optional observability hook — receives every `trace.emit(...)` event |

Discovery is lazy: namespace `ep.load()` is deferred until the first `/_remote` call for that namespace; one broken namespace doesn't block sandbox boot. See [`docs/architecture.md`](docs/architecture.md) and [`docs/namespace-protocol.md`](docs/namespace-protocol.md) for protocol details.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Contributing

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Project conventions in [CLAUDE.md](CLAUDE.md) — read the "组合优于继承 / Composition over inheritance" section.

## License

[MIT License](LICENSE)
