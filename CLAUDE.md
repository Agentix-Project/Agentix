# Project Conventions

## Product context

Agentix targets **agent eval**, **RL rollouts**, and **rollout data
collection** (via `agentix.utils.trace` + `abridge`). Positioning: a friendlier
alternative to HTTP rollout servers such as
[ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)—
integrate with importable callables and `client.remote(fn, ...)`, not
custom `AgentHandler` services. Public slogan: *The universal bridge
between agents and environments.*

## Two Concepts

Agentix has exactly two ideas:

1. **Remote calls** — `c.remote(fn, *args, **kwargs)` calls an
   importable Python function inside a sandbox worker. The target is
   `fn.__module__ + "::" + fn.__qualname__`; args/kwargs travel as a
   single pickle blob and the return value is unpickled host-side.
2. **Bundle** — `agentix build [path]` packages a Python project and
   its declared dependencies into a deploy-ready Docker image. The
   project's `[project].dependencies` defines what modules are
   installed into the runtime venv.

The primary user model is:

```python
from app import run

result = await client.remote(run, input="hello")
```

`import app; await client.remote(app.run, ...)` also works because it
passes the same function object.

## Three Built-In Systems

agentix-core ships **three** independent systems, mapped to three
reserved Socket.IO namespaces:

| Namespace | System  | Public API                                      |
|-----------|---------|-------------------------------------------------|
| `/rpc`    | RPC     | `client.remote(fn, ...)`                        |
| `/trace`  | tracing | `agentix.utils.trace.span(...)` / `trace.Processor` |
| `/log`    | logging | stdlib `logging` (auto-bridged sandbox → host) |

Plugins (`abridge`, future LLM tools, ...) MUST live on their own
namespace `/<package-name>`. Two plugins can never collide because
PyPI package names are globally unique.

## Composition Over Inheritance

Use inheritance only for genuine lifecycle interfaces:
- `SandboxProvider` Protocol for provider backends
- `agentix.Namespace` / `agentix.AsyncClientNamespace` for plugin SIO
  handlers (mirrors `socketio.AsyncClientNamespace`)
- `trace.Processor` for trace sinks

Everywhere else, prefer normal functions, Protocols, composition
objects, or callbacks. A remote target is just a Python callable
serialized by stdlib pickle — there is no base class for user code to
inherit from.

## No Backward Compatibility Shims

This repo is in active design. Breaking changes are fine.

- Rename by deleting the old name, not by accepting both.
- Do not add deprecation warnings.
- Do not leave comments explaining removed behavior.
- Update tests to the current shape; do not preserve tests for removed
  behavior.

## Monorepo Layout

Everything lives in this one repo, wired as a **uv workspace** — the
core, the plugins, and the cookbook examples. Edit any file and it's
live in the shared venv; there is no commit → push → publish cycle for
day-to-day iteration.

```text
Agentix/                       — repo root = workspace root
├── pyproject.toml             — `agentixx` core package + [tool.uv.workspace]
├── uv.lock                    — one lock for the whole workspace
├── agentix/                   — core source (see Systems Map below)
├── tests/                     — core tests
├── plugins/
│   ├── abridge/               — `agentix-bridge` (import `agentix.bridge`)
│   │   ├── pyproject.toml
│   │   ├── agentix/bridge/
│   │   └── tests/
│   ├── providers/
│   │   ├── docker/            — `agentix-provider-docker` → `docker` + `podman`
│   │   ├── daytona/           — `agentix-provider-daytona` → `daytona`
│   │   ├── e2b/               — `agentix-provider-e2b` → `e2b`
│   │   ├── apptainer/         — `agentix-provider-apptainer` → `apptainer`
│   │   └── uv/                — `agentix-provider-uv` → local uv-materialized runtime (no container)
│   └── runtime-basic/         — `agentix-runtime-basic` → `bash` + `files`
└── examples/
    └── run-swe-rollouts/      — batch SWE-bench rollouts via `agentix.runner`
```

`[tool.uv.workspace] members = ["plugins/*"]` — drop a plugin dir under
`plugins/` and it is a workspace member; `uv sync --all-packages`
installs it editable.

Each provider-backend member is a single module that contributes a
sibling into the core `agentix/provider/` namespace (e.g.
`agentix/provider/docker.py`). The dirs carry no `__init__.py` — that
file belongs to the core. The backend is wired in by its
`[project.entry-points."agentix.provider"]`, which the `Registry`
discovers via `importlib.metadata` — so an editable workspace install
makes `from agentix.provider.docker import DockerProvider` work and
`providers().get("docker")` / `providers().get("podman")` resolve (the
string registry powers the CLI; typed code imports the class directly).

Dependency separation is preserved: each member has its own
`pyproject.toml` + dependency list. The core never pulls a plugin's
deps — `openai` belongs to `agentix-bridge`; the E2B/Daytona SDKs
belong to their backend members, not `agentixx`. Members reference each
other with `[tool.uv.sources] <dep> = { workspace = true }` (editable,
no fetch).

`runtime-basic` is a *sandbox-side* member — it ships the `bash` +
`files` namespaces and their `default.nix` data files into the
`agentix build` bundle. The provider members are *host-side*. Both
kinds are ordinary workspace members; the build layer sorts out which
ones land in the bundle.

## Systems Map

```text
agentix/
├── sio.py             — agentix.Namespace + register_namespace (sandbox side)
├── utils/
│   ├── log/           — stdlib logging Handler bridge (sandbox → host)
│   └── trace/         — Trace + Span + SpanEvent + Processor
├── runtime/
│   ├── shared/        — wire types, codec, framing
│   ├── client/        — RuntimeClient (host) + AsyncClientNamespace
│   └── server/        — FastAPI + Socket.IO + worker subprocess
├── provider/          — SandboxProvider Protocol + live Sandbox + backend loader
├── cli/               — `agentix build` + in-container `_assemble`
└── builder/           — in-container builder (flake.nix, Dockerfile, bundle-build.sh)
```

One line per system:

- **sio** — generic pipe-bridged Namespace API. Sandbox plugins
  subclass `agentix.Namespace`; host plugins subclass
  `agentix.AsyncClientNamespace`. Runtime knows zero plugin event names.
- **utils.log** — installs a `logging.Handler` on the worker's root
  logger; every `LogRecord` ships over `/log` and replays on the host's
  logging tree. Zero new API — users write `logger.info(...)` normally.
- **utils.trace** — OTel-style `Trace` + `Span` + `SpanEvent` +
  `Processor`. Worker-side `Processor` ships span lifecycle as events
  on `/trace`; host-side `RuntimeClient` auto-registers a consumer.
- **runtime.shared** — msgpack codec, length-prefixed worker frames,
  pydantic wire models, branded wire ids.
- **runtime.client** — `RuntimeClient.remote(fn, ...)` over Socket.IO
  `/`. `register_namespace(ns)` attaches plugin handlers.
- **runtime.server** — ASGI app launched by the bundle's `/nix/runtime/bootstrap.sh`; owns one worker subprocess,
  resolves import-path callables, dynamic namespace forwarding for
  `/trace`, `/log`, and any plugin `/<package-name>`.
- **provider** — host-side `SandboxProvider` Protocol, the live `Sandbox`
  handle (`await sandbox.remote(fn, ...)`), and backend lookup.
- **cli** — `agentix build [path]` (host) + `agentix.cli.build.closures`
  (in-container closure discovery).
- **builder** — `flake.nix`, `flake.lock`, `Dockerfile`,
  `bundle-build.sh` shipped as wheel data; `agentix build` stages them
  per invocation. (Previously named `nix/`; renamed because the folder
  also ships a `Dockerfile` and a shell script, so "nix" was misleading.)

## Remote Call Implementation

`c.remote(fn, ...)` encodes `fn` as an import-path `RemoteCallable`
(`module::qualname`). Args and kwargs travel as one pickle blob; the
return value comes back as another pickle blob.

```python
# my_project/tasks.py
async def run(seed: int) -> dict:
    ...

# caller
from my_project.tasks import run

result = await client.remote(run, seed=42)
```

Sync functions work too; the invoker awaits only when the result is
awaitable. Side-channel traffic (trace events, log records, plugin
events) rides separate SIO namespaces via `agentix.sio`.

## Plugin Extension via Namespaces

Plugins (e.g. `abridge`) define **two** classes, one per side:

```python
# Sandbox side (runs in the worker subprocess)
import agentix

class MyService(agentix.Namespace):
    namespace = "/my-plugin"

    async def on_request(self, payload):
        # `payload` is whatever the host emitted — auto-unpacked
        result = await do_work(payload)
        await self.emit("request:result", result)

agentix.register_namespace(MyService())

# Host side
class MyHost(agentix.AsyncClientNamespace):
    def __init__(self):
        super().__init__("/my-plugin")

    async def on_request_result(self, data):
        # data is a plain dict — agentix auto-unpacks msgpack
        ...

client = RuntimeClient(url)
client.register_namespace(MyHost())
async with client as c:
    ...
```

Round-trip helper: `await self.request("op", body)` from sandbox auto-
correlates with the host's `op:result` / `op:error` reply event using
a generated `request_id`.

## Bundle Implementation

`agentix build [path]` produces a bundle from one project root.
The build splits along one hard line — **uv owns Python, Nix owns
system binaries; there is no uv2nix.**

The host side is thin: `agentix build` finds the project's git repo,
copies it into a Docker build context, and runs `docker buildx build`.
Every heavy step happens *inside* the container, so the host needs
only `agentixx`, `docker`, and `git` — no project venv, no uv, no Nix.
The whole repo is the context (so a uv-workspace member / cookbook
example can resolve its path dependencies); the project is addressed
by its subpath.

Inside the container (`agentix/builder/bundle-build.sh`):

1. **Toolchain** — `nix build .#toolchain` materializes the
   interpreter + uv into `/nix/store`. Python version comes from the
   project's `requires-python`.
2. **Python deps** — `uv venv /nix/runtime/venv` + `uv sync` install
   the full dependency closure (non-editable). This is a plain uv
   venv; uv natively handles path / git / registry sources, so there
   is no staging puzzle and no build-backend fragility.
3. **System closures** — `python -m agentix.cli.build.closures` discovers
   every system-deps `{ pkgs }: drv` (see below) and stages them into
   `closures/`.
4. **Runtime** — `nix build .#runtime` `symlinkJoin`s the toolchain +
   all closures; the merged tree is placed at `/nix/runtime`.

System-deps closures, two sources:

- **Plugins.** A plugin registers one entry point in the `agentix.nix`
  group; the value names a module that ships a `default.nix` as
  package data. `_assemble` walks the group + `importlib.resources` —
  import-free, follows the dependency graph, every file has provenance
  (which distribution shipped it). No directory scanning.
- **The project.** Declared as `[tool.agentix] nix = "<path>"` — the
  only place a bundle author writes Nix. Optional.

Plugin/project Nix files follow one convention: `{ pkgs }: drv`. The
builder hands every closure the same Nixpkgs revision (pinned in
`agentix/builder/flake.lock`).

Result image layout (`/nix` is what gets mounted):

```text
/nix/store/...               closures: interpreter, uv, system deps
/nix/runtime/venv            the uv venv — all Python deps
/nix/runtime/{bin,lib,...}   symlinkJoin of every closure
```

The two-image runtime: providers overlay `SandboxConfig.bundle`
(the bundle from `agentix build`) onto `SandboxConfig.image` (a
task-specific base) via Docker 25's `--mount type=image,source=…,
target=/nix,subpath=nix,readonly`. No rebuild needed when the agent
moves between task images.

## Wire Protocol

RPC on `/rpc`:

```text
call         {call_id, callable, arguments}
call:result  {call_id, value}    # value is pickle bytes
call:error   {call_id, error}
cancel       {call_id}
```

Plugin namespaces are opaque to the runtime — events and payload
shapes are extension-chosen. The pipe carries `sio_open` (worker
declares a namespace), `sio_emit` (worker → server → broadcast), and
`sio_inbound` (server → worker, forwarded from a host emit).

Errors stay in-band: Socket.IO emits an error event for RPC; plugin
namespaces follow whatever convention the plugin picks (typically
`event:error` with a matching `request_id`).

## Project Management — uv

This project uses **uv** for everything dependency-related:

- Install / sync: `uv sync` (no `pip install`)
- Add a dep: `uv add <pkg>` (writes `pyproject.toml` + locks)
- Add a dev dep: `uv add --dev <pkg>` (goes to `[dependency-groups]`)
- Ad-hoc install into the venv without touching pyproject: `uv pip install`
- Run a tool from the venv: `uv run <cmd>` or `.venv/bin/<cmd>`
- Build wheels/sdists: `uv build`
- Publish: `uv publish` (only when releasing — see below)

Never invoke `pip` directly; `pip install` bypasses the lockfile and
mixes dependency management styles. When you need a runtime dep, the
right answer is `uv add`. When you need a tool just for the current
venv, `uv pip install`.

## Typing — No Bypass

CI runs `pyright` over the whole workspace and **must** stay at zero
errors. If pyright
flags something, **fix the root cause**, do not `# type: ignore`. Common
patterns that lead to ignore-spam and what to do instead:

- **Stubs lie about a decorator's return.** Call the function
  non-decorator style: `obj.on(name, handler)` instead of
  `@obj.on(name)`. The body of `handler` is still type-checked, and the
  registration side-effect happens just the same.
- **`getattr(self, name)` returns `object`.** Either don't go through
  `getattr` (walk the class via `inspect.getmembers` or `__dict__`),
  or relax the declared type to honestly reflect what could come back.
  A `Handler = Callable[[Any], Any]` is more accurate than
  `Callable[[Any], Awaitable[None] | None]` if the function in fact
  accepts any return type.
- **`Protocol` mismatch after refactor.** Update the Protocol; do not
  ignore-suppress at the assignment site.

`type: ignore` is allowed only when the lie is in a *third-party*
type stub that you cannot fix — and even then, prefer pinning the
narrowest comment (`# type: ignore[specific-rule]`) and noting why.

## Development Distribution

The core and the plugins are members of one uv workspace (see Monorepo
Layout). Install every member editable into one venv:

```
uv sync --all-packages --all-extras
```

`--all-packages` is required — without it `uv sync` installs only the
root (`agentixx`) and its deps, not the plugin members.

The repo sits on a slow FUSE mount, so the venv is created on local
disk and symlinked: `UV_PROJECT_ENVIRONMENT=/root/agentix-venv uv sync
… && ln -s /root/agentix-venv .venv`. A venv built directly on the
FUSE path is pathologically slow and occasionally corrupts.

Day-to-day there is **no commit → push → publish cycle** — editing a
file in `agentix/`, `plugins/abridge/`, or `examples/` is immediately
effective for every other member, because they cross-reference via
`[tool.uv.sources] <dep> = { workspace = true }`.

PyPI publishing (`uv build` + `uv publish`) is reserved for real
releases, not iteration.
