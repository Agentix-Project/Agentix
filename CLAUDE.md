# Project Conventions

## Two Concepts

Agentix has exactly two ideas:

1. **Remote calls** — `c.remote(fn, *args, **kwargs)` calls an
   importable Python function inside a sandbox worker. The target is
   `fn.__module__ + "::" + fn.__name__`; the call shape (unary /
   stream / bidi) is detected from the function signature; the return
   value is decoded into `fn`'s return type.
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

## Composition Over Inheritance

Use inheritance only for genuine lifecycle interfaces such as a
deployment backend implementing the `Deployment` Protocol. Everywhere
else, prefer normal functions, Protocols, composition objects, or
callbacks.

A remote target is just a Python callable serialized by stdlib pickle.
There is no base class for user code to inherit from and no marker
Protocol for users to import.

## No Backward Compatibility Shims

This repo is in active design. Breaking changes are fine.

- Rename by deleting the old name, not by accepting both.
- Do not add deprecation warnings.
- Do not leave comments explaining removed behavior.
- Update tests to the current shape; do not preserve tests for removed
  behavior.

Sibling repos (`Agentix-Runtime-Basic`, `Agentix-Deployment-*`,
`agentix-cookbook`) are updated in lockstep with HEAD.

## Systems Map

```text
agentix/
├── runtime/
│   ├── shared/              — wire types, codec, framing, event names
│   ├── client/              — RuntimeClient
│   └── server/              — FastAPI + Socket.IO + worker package
├── deployment/          — Deployment Protocol + backend plugin loader
├── cli/                 — agentix build
└── nix/                 — shipped Nix builder (flake + uv2nix wrapper)
```

One line per system:

- **runtime.shared** — msgpack codec, length-prefixed worker frames,
  Socket.IO event names, pydantic wire models, call-shape helpers, and
  branded wire ids.
- **runtime.client** — `RuntimeClient.remote(fn, ...)`; Socket.IO for unary,
  stream, and bidi; HTTP only for health.
- **runtime.server** — `agentix-server`; owns one runtime worker process,
  invokes pickle-resolved callables, forwards Socket.IO calls, and
  correlates events by `call_id`.
- **deployment** — host-side `Deployment` Protocol and backend lookup.
- **cli** — `agentix build [path]`.
- **nix** — `flake.nix`, `builder.nix`, `wrapper.nix.tmpl` shipped as
  wheel data; `agentix build` stages them per invocation and drives
  `nix build` to produce the runtime image.

## Remote Call Implementation

`c.remote(fn, ...)` serializes `fn` with stdlib pickle and sends that
callable payload to the runtime.

Example:

```python
# my_project/tasks.py
async def run(seed: int) -> dict:
    ...

# caller
from my_project.tasks import run

result = await client.remote(run, seed=42)
```

The Socket.IO payload carries:

```python
{
    "callable_payload": b"...pickle...",
    "display_name": "my_project.tasks::run",
    "shape": "unary",
    "args": [],
    "kwargs": {"seed": 42},
    "call_id": "optional-correlation-key",
}
```

The worker unpickles the callable, validates args with pydantic, calls
the callable, and serializes the result.

## Call Shapes

Three shapes are detected from `fn`'s signature:

- `async def f(...) -> T` -> **unary**
- `async def f(...) -> AsyncIterator[T]: yield ...` -> **stream**
- `async def f(..., inbox: Channel[I]) -> AsyncIterator[T]` -> **bidi**

`c.remote(...)` returns `Unary[T]`, `Stream[T]`, or `Bidi[I, T]`.
Await unary; `async for` over stream and bidi.

Sync functions work for unary too; the invoker awaits only when the
result is awaitable. Streams and bidi require async generators.

## Bundle Implementation

`agentix build [path]` produces a self-contained, distro-portable
runtime image from one project root. No base image, no `FROM`, no `pip
install` inside the build. Everything goes through Nix:

```toml
[project]
name = "my-agent"
version = "0.1.0"
dependencies = [
    "agentixx>=0.1.2",
    "agentix-runtime-basic>=0.1.2",
    "agentix-deployment-docker>=0.1.3",
]
```

The user must also have a `uv.lock` (run `uv lock`). uv2nix consumes
that lock and produces Nix derivations for every Python dep — the
interpreter, agentixx, plugins, the user's project, and transitive
deps all land in `/nix/store/...` with rpath-resolved closures, so the
resulting image runs against any Linux task image (Alpine, RHEL, etc.)
when overlaid via `SandboxConfig.runtime_image`.

Two inputs to the build:

1. **Python side — `pyproject.toml` + `uv.lock`.** uv2nix reads the
   lock; `mkVirtualEnv` materializes a venv with the full closure;
   `/bin/agentix-server` becomes the entry point.

2. **System side — plugin `default.nix` files.** Each plugin may ship
   a `default.nix` next to its Python module (e.g.
   `agentix/bash/default.nix`). `agentix build` discovers them via
   `importlib.resources.files('agentix.<short>') / 'default.nix'`,
   imports each with the shared `pkgs`, and `symlinkJoin`s the results
   into the bundle's `/bin/`. Plugins that need no system binaries can
   skip the file entirely. The user's project may add a `default.nix`
   at its root to declare extra system deps (git, ffmpeg, ...).

Plugin nix expressions follow one convention: `{ pkgs }: drv`. The
builder hands every plugin the same Nixpkgs revision (pinned in
`agentix/nix/flake.lock`), so no per-plugin version drift.

Build flow:

1. Stage a temp dir with `_builder/` (shipped flake), `project/` (user
   source), `plugins/<short>.nix` (discovered plugin nix files), and
   a generated `flake.nix` wrapper.
2. `nix build .#bundle` → uv2nix overlay → `mkVirtualEnv` → joined
   tree → `streamLayeredImage` script.
3. `<result> | docker load` produces a local docker image tagged
   `<name>:<tag>`.

The two-image runtime: deployments overlay `SandboxConfig.runtime_image`
(the bundle from `agentix build`) onto `SandboxConfig.image` (a
task-specific base) via Docker 25's `--mount type=image,source=…,
target=/nix,subpath=nix,readonly` at sandbox-create time. No rebuild.

## Wire Protocol

Unary uses Socket.IO:

```text
unary        {call_id, callable_payload, display_name, shape, args, kwargs}
unary:result {call_id, value}
unary:error  {call_id, error}
```

Stream and bidi use Socket.IO events:

```text
stream       {call_id, callable_payload, display_name, shape, args, kwargs}
stream:item  {call_id, value}
stream:end   {call_id}
stream:error {call_id, error}

bidi:start   {call_id, callable_payload, display_name, shape, args, kwargs}
bidi:in      {call_id, item}
bidi:end_in  {call_id}
bidi:out     {call_id, value}
bidi:end     {call_id}
bidi:error   {call_id, error}
```

Errors stay in-band: Socket.IO emits an error event for unary, stream,
and bidi.
