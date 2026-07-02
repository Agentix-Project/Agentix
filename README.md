<div align="center">

<h1>Agentix</h1>

### The universal bridge between agents and environments.

<p>
Evaluate agents, train them with RL, and collect rollout data across
<strong>any agent</strong> and <strong>any sandbox</strong> — one Python call,
no bespoke microservice per pairing and no changes to the agent.
</p>

[![GitHub Stars](https://img.shields.io/github/stars/Agentix-Project/Agentix?style=flat-square)](https://github.com/Agentix-Project/Agentix)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-online-cc785c?style=flat-square)](https://agentix-project.github.io/Agentix/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

**[Docs](https://agentix-project.github.io/Agentix/)** · **[Quickstart](https://agentix-project.github.io/Agentix/quick-start/)** · **[Cookbook](https://github.com/Agentix-Project/agentix-cookbook)** · **[Roadmap](ROADMAP.md)**

</div>

---

Agentix is the universal bridge between agents and environments — the
single path connecting **any agent**, **any sandbox**, and **any model**
for evaluation, RL training, rollout-data collection, and observability.
It stays small on purpose: two ideas (a *bundle* and a *remote call*) plus
one model-call bridge ([abridge](plugins/abridge/README.md)), with no
heavy stack of disconnected sandbox runners, rollout services, and agent
frameworks to hold together. Five core capabilities:

1. **Drive a sandbox with a Python function** — call any importable
   callable inside the box and get its typed value back; extend the loop
   by writing another function.
2. **Eval/run any agent, in any sandbox, with any model** — abridge
   tunnels the agent's LLM calls to the host and translates
   Anthropic → OpenAI-compatible at the wire (more directions on the
   [roadmap](plugins/abridge/ROADMAP.md)), so an off-the-shelf agent runs
   against whatever provider you have, with no agent changes.
3. **Train any agent as an RL rollout — no code change** — the agent is an
   opaque trajectory producer; Agentix makes no assumption about how it is
   built.
4. **Every model call stamped and traced** — abridge tags each call with
   session/request ids at the transport layer and records prompt,
   completion, and token usage as trace spans; trainer-ready
   token-in/token-out capture is being built on top of this.
5. **Observability for free** — every model call becomes an OTel span,
   exportable to LangSmith / LangFuse / Docent / any OTLP backend with zero
   agent instrumentation.

<table>
<tr>
<td width="50%" valign="top">

#### Any agent

Claude Code · mini-SWE-agent · Qwen Code · your own  
Expose as `async def run(...) -> Result`.

</td>
<td width="50%" valign="top">

#### Any environment

SWE-bench images · custom Docker/Podman · Apptainer · your own backend  
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
git clone https://github.com/Agentix-Project/Agentix && cd Agentix
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
[quickstart](https://agentix-project.github.io/Agentix/quick-start/).

## What you can call

The point of one call surface is that an eval or RL loop wires together
out of the same primitive — the agent, the environment setup, and the
scorer are all just functions you remote-call:

| You have | You expose | You call |
|---|---|---|
| An agent (Claude Code, Qwen Code, …) | `async def run(...) -> RunResult` | `await sandbox.remote(run, ...)` |
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
top of Docker/Podman or Apptainer today, with managed backends on the
roadmap.

| | swe-rex · E2B · Daytona · Harbor | Agentix |
|---|---|---|
| **Reach into the sandbox** | Fixed RPC surface, or shell / `docker exec` + vendor SDK | `await sandbox.remote(fn, ...)` — any importable function |
| **Sandbox logs & stdout** | Scrape command output | stdlib `logging` auto-bridged to the host over `/log` |
| **Observability** | Bring your own | `/trace` spans (OTel-shaped) for every step |
| **Model under test** | Whatever the agent's SDK speaks | [`abridge`](plugins/abridge/README.md) translates Anthropic → OpenAI-compatible at the wire — Claude-speaking agents on OpenAI, OpenRouter, vLLM, your gateway |

**vs. rollout-as-a-service**
([ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)).
ProRL popularized an HTTP server with task-specific handlers and token
trajectories for RL trainers. Agentix shares the decoupling — training
stays separate from rollout execution — with a lighter surface.

| | ProRL-Agent-Server | Agentix |
|---|---|---|
| **Add a new task** | Implement a handler, register it | Write a function, install it |
| **Call a rollout** | HTTP request to the service | `await sandbox.remote(fn, ...)` |
| **Trajectories** | Token-in / token-out over the service API | [`abridge`](plugins/abridge/README.md) stamps session/request ids + per-call trace spans; token-in/token-out capture in development |
| **Sweet spot** | HPC-scale multi-turn RL fleets | Teams wiring eval + RL data without a platform team |

Both designs are powerful at HPC scale. Agentix targets the much larger
set of research and product teams that want `await remote(fn)` with fewer
moving parts.

## What you get

Five capabilities, one primitive underneath each:

### 1 · Drive a sandbox with a Python function

`await sandbox.remote(fn, ...)` runs **any importable Python callable**
inside the sandbox and returns its typed value. No fixed RPC surface to
conform to, no base class to inherit — extend the loop by writing another
function. The agent, the repo setup, the scorer are all just functions
you remote-call. `print`, stdlib `logging`, and OTel-shaped `/trace`
spans from inside the sandbox replay on the host automatically.

### 2 · Eval/run any agent, in any sandbox, with any model

Bring an off-the-shelf agent (Claude Code, mini-SWE-agent, Qwen Code, or
your own), drop it in a backend (Docker/Podman, Apptainer, or your own
`SandboxProvider`; managed backends like Daytona and E2B are stubbed on
the roadmap), and point it at your model.
[`abridge`](plugins/abridge/README.md) tunnels the agent's LLM calls back
to the host and translates **Anthropic → OpenAI-compatible** at the wire —
so a Claude-speaking agent runs against whatever you've got: OpenAI,
OpenRouter, a private vLLM/SGLang, your own gateway. More translation
directions (OpenAI → Anthropic, Gemini) are on the
[abridge roadmap](plugins/abridge/ROADMAP.md). The tunnel carries JSON
request/response bodies and buffers responses (no incremental streaming
yet) — see the [abridge README](plugins/abridge/README.md) for the exact
contract.

### 3 · Turn that same agent into an RL rollout — no code change

Agentix treats the agent as an **opaque trajectory producer** and makes
no assumption about how it's built: single-shot or multi-turn,
deep-thinking loops, hierarchical multi-agent — all opaque internal
logic. Because every completion request crosses the bridge, the exact
context presented to the model at each call is observable on the per-call
trace span, with **zero instrumentation in the agent**. The same run you
evaluate is the run you train on.

### 4 · Every model call stamped and traced

[`abridge`](plugins/abridge/README.md) stamps every model call with a
session id (per rollout) and request id (per call) at the transport
layer — the agent never sees it — and records prompt, completion, and
token usage on the call's trace span. That gives an upstream gateway
everything it needs to group a rollout's calls; trainer-ready
**token-in / token-out** capture is being built on top of this (see the
[abridge roadmap](plugins/abridge/ROADMAP.md)). Nothing to wire into the
agent: the same run you evaluate is the run you train on.

### 5 · Observability, with zero agent instrumentation

Every model call abridge tunnels also becomes an OTel-shaped span — tagged
with GenAI semantic conventions (model, token usage, prompt/completion
content, tool calls) — and fed into the core `/trace` system. Register one
`Processor` ([`agentix-trace-otel`](plugins/trace-otel/README.md)) and the
full agent trajectory exports to **LangSmith, LangFuse, Docent, Phoenix,
or any OTLP backend**. The agent stays pristine — Agentix derives the
spans from the traffic it already bridges, so there is nothing to
instrument.

## Ecosystem

One monorepo, separate packages (`agentixx` and `agentix-runtime-basic`
are on PyPI today; the rest install from source while publishing catches
up). The core is `agentixx`; everything else is an optional plugin under
[`plugins/`](plugins).

| Package | Role |
|---|---|
| [`agentix-runtime-basic`](plugins/runtime-basic/README.md) | `agentix.bash`, file ops, sandbox primitives |
| [`agentix-provider-docker`](plugins/providers/docker) · [`-apptainer`](plugins/providers/apptainer) | Sandbox backends ([`-daytona`](plugins/providers/daytona) · [`-e2b`](plugins/providers/e2b) are placeholders pending integration) |
| [`agentix-runner`](plugins/runner/README.md) | `run_rollouts(...)` — batch eval/rollout orchestration |
| [`agentix-dataset-swe`](plugins/datasets/swebench) | SWE-bench task images + official-harness scoring |
| [`agentix-agent-claude-code`](plugins/agents/claude-code) · [`-mini-swe-agent`](plugins/agents/mini-swe-agent) · [`-qwen-code`](plugins/agents/qwen-code) | Agent adapters |
| [`agentix-bridge`](plugins/abridge/README.md) | Model translation + rollout → RL buffer capture (abridge) |
| [`agentix-trace-otel`](plugins/trace-otel/README.md) | Export `/trace` spans to any OTLP backend |

Drop a directory under `plugins/` and it becomes a workspace member;
`uv sync --all-packages` installs it editable.

## Development

```bash
git clone https://github.com/Agentix-Project/Agentix
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

- [Docs](https://agentix-project.github.io/Agentix/) · [Quickstart](https://agentix-project.github.io/Agentix/quick-start/)
- [Architecture](https://agentix-project.github.io/Agentix/architecture/) · [CLI](https://agentix-project.github.io/Agentix/cli/) · [Plugins](https://agentix-project.github.io/Agentix/plugins/)
- [Architecture (source)](ARCHITECTURE.md) · [Roadmap](ROADMAP.md)

<div align="center">
<sub>MIT licensed · built on <a href="https://docs.astral.sh/uv/">uv</a> workspaces</sub>
</div>
