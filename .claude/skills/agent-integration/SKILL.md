---
name: agent-integration
description: Use this skill when the user wants to add a new agent closure to Agentix-Agents-Hub, a new dataset/verifier closure to Agentix-Datasets, or port an existing integration from llm-agents.nix (numtide/llm-agents.nix) into Agentix's typed Python closure convention. Triggers include phrases like "add <name> to agents-hub", "port <name> from llm-agents.nix", "wrap the <cli> CLI as an Agentix closure", "integrate <tool> as a closure service", or any mention of building/adapting a new CLI-driven agent or evaluation harness to run inside an Agentix sandbox.
version: 1.0.0
---

# Agent Integration

Guide for packaging an external CLI-based agent (or a dataset/verifier) into an **Agentix typed Python closure** — a Docker image that ships a Python package the runtime imports in-process and exposes via `RuntimeClient.remote(<stub>, ...)` calls.

Upstream reference for Nix binary packaging:

- **llm-agents.nix** — how to pin versions and produce a hermetic `bin/<cli>`. Path: `../llm-agents.nix` (relative to the Agentix repo root).

CLI invocation logic (argv, env, flag flow) typically comes from the upstream agent's own docs or any existing wrapper you have handy; this skill describes how to land it in the Agentix closure shape.

> **Current layout (this monorepo, today).** Integrations now live *in-repo* as
> uv-workspace members under `plugins/agents/<name>/` and
> `plugins/datasets/<name>/`, following `docs/integrate-agent.mdx` and
> `docs/integrate-dataset.mdx`. Each is a plain Python package
> (`agentix.agents.<name>` / `agentix.plugins.datasets.<name>`) that exports a
> normal `run` / `score` callable invoked via `client.remote(fn, ...)`, with an
> optional `default.nix` (`{ pkgs }: drv`) registered through
> `[project.entry-points."agentix.nix"]`. Working references to copy:
> `plugins/agents/claude-code/` and `plugins/datasets/swebench/`.
>
> The self-contained closure ABI described below (`agentix_closures.<name>`,
> `manifest.json`, `Dispatcher`, per-closure `Dockerfile`, separate
> `Agentix-Agents-Hub` / `Agentix-Datasets` repos) is the **planned post-split
> target** for when integrations graduate into their own repos. When working
> inside this monorepo, follow the `plugins/` convention above and treat the
> sections below as the direction of travel.

## When to activate

- "add claude-code / aider / opencode / gemini-cli / qwen-code ... to agents-hub"
- "port <name> from llm-agents.nix" — llm-agents.nix is the source of truth for the derivation.
- "wrap the <X> CLI as an Agentix closure"
- "add a new dataset closure (e.g. humaneval, mbpp, livecodebench) to Agentix-Datasets"
- The user references llm-agents.nix as a reference / starting point for a new closure.

Do **not** use this skill for tasks that only touch `Agentix` core (runtime server, deployment, dispatcher). Those are core-substrate edits, not integrations.

## The Agentix closure ABI (non-negotiable)

The final image must satisfy:

- `VOLUME /nix`
- `/nix/store/<hash>-*/` — content-addressed Nix deps for everything the closure needs.
- `/nix/entry/python/agentix_closures/<name>/` — Python package the runtime imports.
  - `__init__.py` — typed stubs (signatures only; body raises `NotImplementedError`).
  - `_impl.py` — real implementations.
  - `_register.py` — `def register() -> Dispatcher` that binds each stub to its impl.
- `/nix/entry/manifest.json` — `ClosureManifest` with `abi == AGENTIX_CLOSURE_ABI` and `package = "agentix_closures.<name>"`. This file is what marks `/mnt/<dir>` as a closure; without it the runtime ignores the mount.
- Optional: `/nix/entry/bin/<cli>` — native binaries the impl shells out to.

There is no `bin/start`, no UDS, no FastAPI app inside the closure. Caller-side typing flows from `RuntimeClient.remote(<stub>, ...)`, where `<stub>` is a regular Python callable imported from `agentix_closures.<name>`.

See `docs/closure-protocol.md` in Agentix for the full protocol.

## Each closure is self-contained

Every closure directory is an **independently maintained unit**. This is deliberately the opposite of llm-agents.nix's monorepo-with-blueprint organization.

- Own `Dockerfile`. No shared template file referenced across closures.
- Own `default.nix`. No `import ../<other>/` across closure boundaries, no shared `lib/` or `overlays/` at the repo root.
- Own `pyproject.toml`. The Python package is pip-installable on its own (`agentix-closure-<name>`).
- Own version pins. Two closures can run different versions of the same CLI — that's fine.
- Own lifecycle: build, test, bump, ship without touching siblings.

A single `default.nix` per closure is preferred (inline the binary derivation as a `let` binding); only split into a separate `package.nix` when the file gets unwieldy. Reference material in the Agentix core repo (e.g. `tests/closure-docker/Dockerfile`) is meant to be **copy-pasted as a starting point**, not referenced at build time.

## Target closure layout

Every migrated closure lives at `Agentix-Agents-Hub/<name>/` (agents) or `Agentix-Datasets/<name>/` (datasets) with this shape:

```
<name>/
├── pyproject.toml                          # package metadata: name = "agentix-closure-<name>"
├── agentix_closures/
│   └── <name>/
│       ├── __init__.py                     # typed stubs
│       ├── _impl.py                        # real implementation
│       └── _register.py                    # register() -> Dispatcher
├── manifest.json                           # ClosureManifest (abi, package, ...)
├── default.nix                             # binary derivation(s) + buildPythonPackage + symlinkJoin
└── Dockerfile                              # closure image: nix-build → /export → final layer
```

`__init__.py`, `_impl.py`, `_register.py`, `manifest.json`, `default.nix`, `Dockerfile`, `pyproject.toml` are all mandatory.

## Source mapping

**The governing principle: locate → extract, don't copy.** Upstream packaging carries a lot of machinery we don't use (Blueprint flakes, overlays, custom lib hooks, update scripts). For each new closure you only need a few facts out of each source file — everything else goes straight in the bin.

### Lift the CLI invocation into `_impl.py`

Whatever your reference is (the upstream CLI's README, an existing harness, your own notes), boil it down to three things and forget the rest:

- **argv** — the exact subprocess argument list (e.g. `claude -p <prompt> --output-format=stream-json --permission-mode=bypassPermissions --print --`). Drop it literally into `_impl.py`'s `subprocess` call.
- **fixed env** — variables the agent hardcodes regardless of caller (`DISABLE_AUTOUPDATER=1`, `IS_SANDBOX=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, …). Set these inside `_impl.run` when building the subprocess env.
- **caller-provided env** — variables the agent reads at runtime (`ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, model selection, …). Expose as a typed `env: dict[str, str] | None = None` keyword argument on the stub; `_impl.run` layers it over `os.environ`.

Ignore everything else — Pydantic config schemas, trajectory parsing, multi-backend abstractions, `apt-get`/`npm install` setup logic (Nix replaces them).

**Skills / memory / MCP registration**: when the agent normally pulls these from host-side paths (`~/.claude/skills`, etc.), expose them as **typed dict parameters on the stub**, not filesystem paths. A closure has no "host" — the orchestrator ships content inline. Pattern:

```python
def run(
    instruction: str,
    *,
    skills: dict[str, str] | None = None,           # {relpath: content}
    memory: dict[str, str] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
) -> RunResult: ...
```

`_impl.py` materialises these under `$CLAUDE_CONFIG_DIR` (or the agent's config dir) right before invoking the CLI. Reject keys that are absolute or contain `..`. Skip the whole mechanism if your agent doesn't use any of it.

### From llm-agents.nix — lift the source descriptor

Open `llm-agents.nix/packages/<name>/{package.nix,hashes.json}` and lift just these:

- **source descriptor** (URL template + platform map, `fetchFromGitHub` args, or npm tarball URL).
- **version + hash values** (merge `hashes.json` contents into the derivation — no side-car).
- **wrap env / PATH additions** from `postFixup` / `postInstall` (only those the agent actually needs).

Ignore the rest: Blueprint flake auto-discovery, `overlays/default.nix`, custom hooks (`wrapBuddy`, `versionCheckHook`, `versionCheckHomeHook`, `fetchNpmDepsWithPackuments` — drop these or substitute plain nixpkgs equivalents), `__noChroot`, `passthru.category`, `update.py`.

Three derivation shapes cover ~95% of cases. Use `finalAttrs` so the version appears once.

**Binary blob** (claude-code, gemini-cli, …):
```nix
stdenv.mkDerivation (finalAttrs: {
  pname = "<name>"; version = "2.1.114";
  src = fetchurl {
    url  = "https://<cdn>/${finalAttrs.version}/linux-x64/<binary>";
    hash = "sha256-...";
  };
  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];
  installPhase = ''install -Dm755 $src $out/bin/<binary>'';
  postFixup = ''wrapProgram $out/bin/<binary> --set DISABLE_AUTOUPDATER 1'';
})
```

**npm package** (pi, opencode, …):
```nix
buildNpmPackage (finalAttrs: {
  pname = "<name>"; version = "0.67.68";
  src = fetchurl {
    url  = "https://registry.npmjs.org/<pkg>/-/<pkg>-${finalAttrs.version}.tgz";
    hash = "sha256-...";
  };
  npmDepsHash = "sha256-...";
  dontNpmBuild = true;
})
```

**Rust from source** (code, agent-browser, …):
```nix
rustPlatform.buildRustPackage (finalAttrs: {
  pname = "<name>"; version = "0.6.93";
  src = fetchFromGitHub {
    owner = "..."; repo = "..."; tag = "v${finalAttrs.version}";
    hash = "sha256-...";
  };
  cargoHash = "sha256-...";
})
```

## Migration playbook

### 1. Locate source files

```
llm-agents.nix/packages/<name>/{package.nix,hashes.json}
```

Plus whatever reference describes the CLI's invocation (the agent's README, an existing wrapper, etc.). Read first, then lift 2-3 specific pieces (see "source mapping" above) and stitch them together — don't copy verbatim.

### 2. `__init__.py` — typed stubs

Define the request/response dataclasses and the function signatures. Function bodies raise `NotImplementedError` — they're never called on the caller side.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    patch: str

def run(
    instruction: str,
    *,
    workdir: str = "/testbed",
    timeout: float = 600,
    model: str | None = None,
    env: dict[str, str] | None = None,
    # … other agent-specific knobs
) -> RunResult:
    """Doc for callers — what the agent does, what each arg means."""
    raise NotImplementedError("call via RuntimeClient.remote(<name>.run, ...)")
```

The signature is the contract on the caller side — mypy / pyright validate
host code against it. Over the wire, args and return values travel as
pickle blobs today; the worker does not run pydantic validation on them.

### 3. `_impl.py` — real bodies

Same signature, real body. Subprocess the CLI, build env, capture output, return `RunResult`.

```python
import asyncio, os
from . import RunResult

async def run(instruction: str, *, workdir: str = "/testbed", ...) -> RunResult:
    env = _build_env(...)
    argv = ["<cli>", "--foo", "bar", "-p", instruction]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=PIPE, stderr=PIPE, cwd=workdir, env=env,
    )
    out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return RunResult(
        exit_code=proc.returncode or 0,
        stdout=out_b.decode(),
        stderr=err_b.decode(),
        patch=await _collect_patch(workdir),
    )
```

`_impl.py` can be sync or async — the dispatcher awaits awaitable returns.

### 4. `_register.py` — bind stubs to impls

```python
from agentix.dispatch import Dispatcher
from . import run
from ._impl import run as _run_impl

def register() -> Dispatcher:
    d = Dispatcher()
    d.bind(run, _run_impl)
    return d
```

Pure function, no globals. Runtime calls it once on startup.

### 5. `manifest.json`

```json
{
  "abi": 1,
  "name": "<name>",
  "version": "0.1.0",
  "kind": "agent",
  "package": "agentix_closures.<name>",
  "description": "Short blurb"
}
```

### 6. `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agentix-closure-<name>"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []  # closures share the runtime's interpreter; keep this empty

[tool.hatch.build.targets.wheel]
packages = ["agentix_closures/<name>"]
```

### 7. `default.nix`

Composition: binary derivation + `buildPythonPackage` for the Python wheel + `symlinkJoin` to assemble the final tree.

```nix
{ pkgs ? import <nixpkgs> {} }:
let
  binary = pkgs.stdenv.mkDerivation (finalAttrs: {
    pname = "<name>"; version = "X.Y.Z";
    src = pkgs.fetchurl { url = "..."; hash = "sha256-..."; };
    # ... installPhase, postFixup, etc.
  });
  python = pkgs.python312;
  closurePkg = python.pkgs.buildPythonPackage {
    pname = "agentix-closure-<name>"; version = "0.1.0"; format = "pyproject";
    src = ./.;
    nativeBuildInputs = [ python.pkgs.hatchling ];
    doCheck = false;
  };
in
pkgs.symlinkJoin {
  name = "agentix-closure-<name>";
  paths = [
    binary
    (pkgs.runCommand "<name>-python" { } ''
      mkdir -p $out/python
      cp -r ${closurePkg}/lib/python${python.pythonVersion}/site-packages/agentix_closures $out/python/
      cp ${./manifest.json} $out/manifest.json
    '')
  ];
}
```

### 8. `Dockerfile`

Copy `Agentix/tests/closure-docker/Dockerfile` into the closure dir as a starting point. **Context = the closure dir itself** (`<name>/`) — closures are self-contained.

If the closure needs a flake input (e.g. `numtide/llm-agents.nix` for the `claude` binary), put a local `flake.nix` + `flake.lock` *inside the closure dir* and build with `nix build .#<name>` inside the Dockerfile's builder stage.

Labels worth adding: `LABEL org.agentix.closure=1`, `LABEL org.agentix.closure.kind=agent` (or `dataset`), `LABEL org.agentix.closure.name=<name>`.

### 9. No registration step

There's no repo-level registry to update. Each closure's identity is its built Docker image ref plus its `manifest.package`. Callers reference the image by tag in `SandboxConfig(closures=[...])` and import the typed stub: `from agentix_closures import <name>`.

### 10. Build and smoke-test

```bash
docker build -t agentix-hub/<name>:0.1.0 <name>/    # context = the closure dir

# In a tiny script:
import asyncio
from agentix import RuntimeClient, SandboxConfig
from agentix.provider.docker import DockerProvider
from agentix_closures import <name>

async def check():
    cfg = SandboxConfig(
        image="ubuntu:24.04",
        runtime="agentix/runtime:0.1.0",
        closures=["agentix-hub/<name>:0.1.0"],
    )
    async with DockerProvider().session(cfg) as sb:
        async with RuntimeClient(sb.runtime_url) as c:
            print(await c.closures())
            result = await c.remote(<name>.run, instruction="...")
            print(result)
asyncio.run(check())
```

Verify the typed return shape, exit code, stdout, patch. Then add an end-to-end test against a known task (e.g. a SWE-bench instance) before declaring done.

## Common pitfalls

- **Body in the stub** — `__init__.py` functions must `raise NotImplementedError`, not contain real logic. Real bodies belong in `_impl.py`. A stub with a working body will silently run on the caller side instead of being dispatched.
- **Imported deps in `__init__.py`** — keep stub imports minimal (stdlib + dataclasses + typing). Heavy imports (subprocess wrappers, fastapi, etc.) belong in `_impl.py` so caller-side `from agentix_closures import <name>` stays cheap and dependency-free.
- **Agent binary not on PATH inside impl** — your derivation's `symlinkJoin` must produce `bin/<cli>` so it lands at `/mnt/<dir>/entry/bin/<cli>`. `nix-build && ls result/bin/` to verify. The impl invokes the CLI by name; the runtime sandbox makes `entry/bin` resolvable via `paths_from`.
- **Trajectory overload** — resist piping the agent's full step-by-step trajectory through `run()`. Return `{exit_code, stdout, stderr, patch}` and let callers opt in to richer formats via a separate stub method.
- **System deps at runtime** — don't add `apt-get install git` in the Dockerfile runtime stage. Put `pkgs.git` in the derivation. The task image usually has `git` already.
- **Closure Python deps** — keep `pyproject.toml`'s `dependencies = []`. Closures share the runtime's interpreter (`pydantic` is already there). Adding a third-party Python dep to a closure adds risk of collision with other closures' deps.
- **Flake lives in the closure dir** — if the closure needs a flake input, put the `flake.nix` / `flake.lock` in that closure's own directory and build with `nix build .#<name>` inside the Dockerfile.
- **Version bumps** — llm-agents.nix uses `hashes.json` + `update.py` for upgrades. We inline everything; bumping a version is a manual edit of the `version = ...` and matching hash lines.

## Datasets variation

For `Agentix-Datasets/<name>/`, the layout is the same but the stubs export `setup` and `verify` instead of `run`:

```python
@dataclass
class SetupResult:
    instruction: str
    workdir: str
    instance_id: str

@dataclass
class VerifyResult:
    passed: bool
    reason: str

def setup(instance_id: str) -> SetupResult: ...
def verify(patch: str, instance_id: str) -> VerifyResult: ...
```

There's rarely an external binary to vendor — most datasets only need Python + test tooling. `default.nix` depends on `<nixpkgs>` directly.

See `Agentix-Datasets/swebench/` for a working reference.

## Worked example pointer

- llm-agents.nix source: `llm-agents.nix/packages/claude-code/{package.nix,hashes.json}`
- In-repo reference (today): `plugins/agents/claude-code/` (agent) and `plugins/datasets/swebench/` (dataset)
- Post-split target: `Agentix-Agents-Hub/claude-code/`

When migrating a *new* agent, diff your work against `plugins/agents/claude-code/` — it's the canonical reference for the current `plugins/` integration shape.
