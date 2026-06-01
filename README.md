<div align="center">

<h1>Agentix</h1>

### The universal bridge between agents and environments.

<p>
Evaluate agents, run RL rollouts, and collect rollout data across
<strong>any agent</strong> and <strong>any sandbox</strong> — one API, no
bespoke microservice per pairing.
</p>

[![GitHub Stars](https://img.shields.io/github/stars/Agentiix/Agentix?style=flat-square)](https://github.com/Agentiix/Agentix)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-agentiix.github.io-cc785c?style=flat-square)](https://agentiix.github.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

**[Docs](https://agentiix.github.io/)** · **[Quickstart](https://agentiix.github.io/quickstart)** · **[Cookbook](https://github.com/Agentiix/agentix-cookbook)** · **[Roadmap](ROADMAP.md)**

</div>

---

<table>
<tr>
<td width="50%" valign="top">

#### Any agent

Claude Code · Codex · Aider · OpenHands · your own  
Expose as `async def run(...) -> Result`.

</td>
<td width="50%" valign="top">

#### Any environment

SWE-bench images · custom Docker · Daytona · E2B · your own backend  
Pick a sandbox — or bring your own.

</td>
</tr>
<tr>
<td colspan="2" align="center">

⇣ &nbsp; **bridged by** &nbsp; ⇣

```python
await sandbox.remote(fn, *args, **kwargs)
```

</td>
</tr>
</table>

## Two ideas

Agentix is small on purpose. The whole framework is two operations:

| | You write | You get |
|---|---|---|
| **Bundle** | `agentix build [path]` | A deploy-ready image with your code and its dependencies |
| **Remote call** | `await sandbox.remote(fn, ...)` | The return value of `fn`, executed *inside* the sandbox |

`fn` is any importable Python callable — an agent, a shell helper, a
scorer, or a whole multi-step rollout. Args travel in, the typed return
value comes back out. There is no fixed RPC surface to conform to and no
base class for your code to inherit.

```python
from app import run

result = await sandbox.remote(run, input="hello")
```

Side traffic rides along automatically: stdlib `logging` from inside the
sandbox replays into your host logs, and OTel-shaped `/trace` spans
capture every step — ready for eval dashboards and RL buffers.

## Quickstart

```bash
pip install agentixx agentix-runtime-basic agentix-provider-docker
```

Build a bundle once (takes a few minutes), then every remote call is
seconds. From [`examples/hello-world`](examples/hello-world/README.md):

```bash
cd examples/hello-world
uv sync
uv run agentix build . --output dist/hello-world.bundle.tar
BUNDLE=$(uv run agentix deploy docker dist/hello-world.bundle.tar --format json | jq -r .bundle)
uv run python main.py --bundle "$BUNDLE"
```

The host code is just provider → session → remote call:

```python
from agentix.bash import run
from agentix.provider.base import SandboxConfig
from agentix.provider.docker import DockerProvider

config = SandboxConfig(image="python:3.13-slim", bundle=BUNDLE)

async with DockerProvider().session(config) as sandbox:
    result = await sandbox.remote(run, command="echo hello from $(uname -a)")
```

Build a cross-arch bundle by passing `--platform linux/amd64` to both
`agentix build` and `agentix deploy`. Full walkthrough:
[quickstart](https://agentiix.github.io/quickstart).

## What you can call

The point of one call surface is that an eval or RL loop wires together
out of the same primitive — the agent, the environment setup, and the
scorer are all just functions you remote-call:

| You have | You expose | You call |
|---|---|---|
| An agent (Claude Code, Codex, OpenHands, …) | `async def run(...) -> RunResult` | `await sandbox.remote(run, ...)` |
| Shell, files, repo setup | `async def run(command: str) -> BashResult` | `await sandbox.remote(bash_run, ...)` |
| A benchmark or reward model | `async def score(...) -> Score` | `await sandbox.remote(score, ...)` |

[`examples/run-swe-rollouts`](examples/run-swe-rollouts/README.md) is the
full loop end to end: sandbox agent run → patch extraction → SWE-bench
harness score → one rollout log per instance.

## How it compares

**vs. sandbox runners** ([swe-rex](https://github.com/SWE-agent/SWE-ReX),
E2B, Daytona, Harbor). A runner hands you a box and a *fixed* way to reach
into it — a predefined RPC surface, or "run a shell / `docker exec`
command" plus a vendor SDK. Anything richer means squeezing your logic
through that narrow hole. Agentix inverts it: the bundle installs your
real Python, and `sandbox.remote(fn, ...)` calls **any importable
function** and returns its typed value. A backend decides *where* the box
runs; Agentix decides *what you can call inside it* — so you layer it on
top of Docker, E2B, or Daytona.

| | swe-rex · E2B · Daytona · Harbor | Agentix |
|---|---|---|
| **Reach into the sandbox** | Fixed RPC surface, or shell / `docker exec` + vendor SDK | `await sandbox.remote(fn, ...)` — any importable function |
| **Sandbox logs & stdout** | Scrape command output | stdlib `logging` auto-bridged to the host over `/log` |
| **Observability** | Bring your own | `/trace` spans (OTel-shaped) for every step |
| **Model under test** | Whatever the agent's SDK speaks | [`abridge`](plugins/abridge/README.md) translates Claude ⇄ OpenAI ⇄ Gemini — any agent on any model |

**vs. rollout-as-a-service**
([ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)).
ProRL popularized an HTTP server with task-specific handlers and token
trajectories for RL trainers. Agentix shares the decoupling — training
stays separate from rollout execution — with a lighter surface.

| | ProRL-Agent-Server | Agentix |
|---|---|---|
| **Add a new task** | Implement a handler, register it | Write a function, install it |
| **Call a rollout** | HTTP request to the service | `await sandbox.remote(fn, ...)` |
| **Trajectories** | Token-in / token-out over the service API | Captured by [`abridge`](plugins/abridge/README.md) as rollout logs |
| **Sweet spot** | HPC-scale multi-turn RL fleets | Teams wiring eval + RL data without a platform team |

Both designs are powerful at HPC scale. Agentix targets the much larger
set of research and product teams that want `await remote(fn)` with fewer
moving parts.

## What you get

- **One API for everything.** Agent, tool, or scorer — the same
  `await sandbox.remote(fn, ...)`.
- **Bundles from a normal Python project.** `agentix build` reads
  `pyproject.toml`; an optional `default.nix` adds system binaries.
- **Backends you choose.** Local Docker/Podman, Daytona, E2B, Apptainer,
  or your own `SandboxProvider`.
- **Sandbox logs on the host.** `print` and stdlib `logging` from any
  remote call replay into your host logging tree over `/log` — no
  scraping command output.
- **Tracing built in.** OTel-shaped `/trace` spans for every step, the
  same across agents and environments; ship them anywhere with
  [`agentix-trace-otel`](plugins/trace-otel/README.md).
- **Any model behind any agent.** [`abridge`](plugins/abridge/README.md)
  translates between Claude, OpenAI, and Gemini, so an agent that speaks
  one provider can be evaluated against any model — and the host captures
  the trajectory (token-in / token-out) for RL.

## Ecosystem

One monorepo, separate PyPI packages. The core is `agentixx`; everything
else is an optional plugin under [`plugins/`](plugins).

| Package | Role |
|---|---|
| [`agentix-runtime-basic`](plugins/runtime-basic/README.md) | `agentix.bash`, file ops, sandbox primitives |
| [`agentix-provider-docker`](plugins/providers/docker) · [`-daytona`](plugins/providers/daytona) · [`-e2b`](plugins/providers/e2b) · [`-apptainer`](plugins/providers/apptainer) | Sandbox backends |
| [`agentix-runner`](plugins/runner/README.md) | `run_rollouts(...)` — batch eval/rollout orchestration |
| [`agentix-dataset-swe`](plugins/datasets/swebench) | SWE-bench task images + official-harness scoring |
| [`agentix-agent-claude-code`](plugins/agents/claude-code) · [`-mini-swe-agent`](plugins/agents/mini-swe-agent) · [`-qwen-code`](plugins/agents/qwen-code) | Agent adapters |
| [`agentix-bridge`](plugins/abridge/README.md) | Model translation + rollout → RL buffer capture (abridge) |
| [`agentix-trace-otel`](plugins/trace-otel/README.md) | Export `/trace` spans to any OTLP backend |

Drop a directory under `plugins/` and it becomes a workspace member;
`uv sync --all-packages` installs it editable.

## Development

```bash
git clone https://github.com/Agentiix/Agentix
cd Agentix
uv sync --all-packages --all-extras
uv run pytest
uv run ruff check agentix/ tests/
```

This repo is a **uv workspace** — core, plugins, and examples share one
lockfile, so editing any member is live in the shared venv with no
publish cycle. See [ARCHITECTURE.md](ARCHITECTURE.md) for how bundles and
remote calls work under the hood.

## Links

- [Docs](https://agentiix.github.io/) · [Quickstart](https://agentiix.github.io/quickstart)
- [Remote calls](https://agentiix.github.io/concepts/remote-calls) · [Bundles](https://agentiix.github.io/concepts/bundles)
- [Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md)

<div align="center">
<sub>MIT licensed · built on <a href="https://docs.astral.sh/uv/">uv</a> workspaces</sub>
</div>
