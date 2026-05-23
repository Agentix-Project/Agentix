<div align="center">

<h1>Agentix</h1>

### The universal bridge between agents and environments.

<p>
Train, evaluate, and collect rollouts across <strong>any agent</strong> and
<strong>any sandbox</strong> — one API, no bespoke microservice per pairing.
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
await client.remote(fn, *args, **kwargs)
```

</td>
</tr>
</table>

## 30 seconds in

```python
from agentix import RuntimeClient, SandboxConfig, session
from agentix.bash import run
from agentix.deployment.docker import DockerDeployment

config = SandboxConfig(image="python:3.13-slim", bundle="hello-agentix:0.1.0")

async with session(DockerDeployment(), config) as sandbox:
    async with RuntimeClient(sandbox.runtime_url) as client:
        result = await client.remote(run, command="echo hello from $(uname -a)")
```

That's the whole loop. **Bundle** what runs inside the box. **Remote-call**
agents, tools, and scorers as ordinary callables. **Capture** trajectories
with [`abridge`](https://github.com/Agentiix/abridge) for eval and RL.

## Three primitives, one bridge

<table>
<tr><th>Primitive</th><th>You do</th><th>You get</th></tr>
<tr>
<td><strong>Bundle</strong></td>
<td><code>agentix build [path]</code></td>
<td>A deploy-ready image with your code and dependencies</td>
</tr>
<tr>
<td><strong>Remote call</strong></td>
<td><code>await client.remote(fn, ...)</code></td>
<td>Return value of <code>fn</code>, executed inside the sandbox</td>
</tr>
<tr>
<td><strong>Rollout data</strong></td>
<td><code>agentix.trace</code> + <code>abridge</code></td>
<td>Per-rollout logs ready for eval and RL buffers</td>
</tr>
</table>

## Why Agentix exists

Agent **eval**, **RL training**, and **rollout data collection** usually mean
the same bespoke glue: wrap each agent CLI, fork each benchmark harness, bolt
on tracing, then rewrite everything when sandboxes or trainers change.

Agentix collapses that into one rule:

> If your bundle has it, your orchestrator can call it.

<table>
<tr><th>You have</th><th>You expose</th><th>You call</th></tr>
<tr>
<td>An agent (Claude Code, Codex, OpenHands, …)</td>
<td><code>async def run(...) -> RunResult</code></td>
<td><code>await client.remote(run, ...)</code></td>
</tr>
<tr>
<td>Shell, files, repo setup</td>
<td><code>async def run(command: str) -> BashResult</code></td>
<td><code>await client.remote(bash_run, ...)</code></td>
</tr>
<tr>
<td>A benchmark or reward model</td>
<td><code>async def score(...) -> Score</code></td>
<td><code>await client.remote(score, ...)</code></td>
</tr>
</table>

End-to-end loop in [`examples/eval-cc-swe`](examples/eval-cc-swe/README.md):
sandbox agent run → patch extraction → harness score → rollout log per
instance.

## Compared to rollout-as-a-service

[ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)
popularized **rollout-as-a-service**: an HTTP server with task-specific
handlers and token trajectories for RL trainers. Agentix shares the same
decoupling — training stays separate from rollout execution — with a much
lighter surface for the user:

<table>
<tr><th></th><th>ProRL-Agent-Server</th><th>Agentix</th></tr>
<tr><td><strong>Add a new task</strong></td><td>Implement a handler, register it</td><td>Write a function, install it</td></tr>
<tr><td><strong>Call a rollout</strong></td><td>HTTP request to the service</td><td><code>await client.remote(fn, ...)</code></td>
</tr>
<tr><td><strong>Trajectories</strong></td><td>Token-in / token-out over the service API</td><td>Captured by abridge as rollout logs</td></tr>
<tr><td><strong>Sweet spot</strong></td><td>HPC-scale multi-turn RL fleets</td><td>Teams wiring eval + RL data without a platform team</td></tr>
</table>

Rollout-as-a-service is powerful. So is `await remote(fn)` — with fewer
moving parts for most research and product teams.

## What you get

- **One API for everything.** Run an agent, a tool, or a scorer with the
  same `await client.remote(fn, ...)`.
- **Bundles from a normal Python project.** `agentix build` reads
  `pyproject.toml`; an optional `default.nix` adds system binaries.
- **Backends you choose.** Local Docker, Daytona, E2B, or your own.
- **Out-of-the-box tracing & observability.** Trajectory capture works the
  same across agents and environments — ready for eval and RL buffers.

## Quickstart

From [`agentix-cookbook/examples/hello-agentix`](https://github.com/Agentiix/agentix-cookbook/tree/main/examples/hello-agentix):

```bash
cd examples/hello-agentix
uv sync
uv run agentix build . --name hello-agentix --format oci-image
uv run python run.py
```

Cross-arch sandboxes:

```bash
uv run agentix build . --name hello-agentix --format oci-image --platform linux/amd64
```

Full walkthrough: [quickstart](https://agentiix.github.io/quickstart).

## Ecosystem

<table>
<tr><th>Repo</th><th>Role</th></tr>
<tr><td><a href="https://github.com/Agentiix/Agentix-Runtime-Basic">Agentix-Runtime-Basic</a></td><td><code>bash</code>, file ops, sandbox primitives</td></tr>
<tr><td><a href="https://github.com/Agentiix/Agentix-Deployment-Docker">Agentix-Deployment-Docker</a></td><td>Local Docker backend</td></tr>
<tr><td><a href="https://github.com/Agentiix/Agentix-Deployment-Daytona">Agentix-Deployment-Daytona</a> · <a href="https://github.com/Agentiix/Agentix-Deployment-E2B">E2B</a></td><td>Hosted sandbox backends</td></tr>
<tr><td><a href="https://github.com/Agentiix/agentix-cookbook">agentix-cookbook</a></td><td>Agent and benchmark recipes</td></tr>
<tr><td><a href="https://github.com/Agentiix/abridge">abridge</a></td><td>Rollout → RL buffer bridge</td></tr>
</table>

## Development

```bash
git clone https://github.com/Agentiix/Agentix
cd Agentix
uv sync --all-packages --all-extras
uv run pytest
uv run ruff check agentix/ tests/
```

This repo is a **uv workspace** — core, plugins, and examples share one
lockfile.

## Links

- [Docs](https://agentiix.github.io/) · [Quickstart](https://agentiix.github.io/quickstart)
- [Remote calls](https://agentiix.github.io/concepts/remote-calls) · [Bundles](https://agentiix.github.io/concepts/bundles)
- [Roadmap](ROADMAP.md)

<div align="center">
<sub>MIT licensed · built on <a href="https://docs.astral.sh/uv/">uv</a> workspaces</sub>
</div>
