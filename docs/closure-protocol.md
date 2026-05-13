# Closure Protocol (v0.1.0)

A **closure** is a Docker image that ships a Python package the Agentix runtime imports in-process. Calls are typed Python (`await c.remote(my_closure.fn, x=1)`), not arbitrary HTTP — the runtime exposes a single dispatch endpoint and routes by Python module path.

## Image convention

A closure image MUST:

1. Declare `VOLUME /nix` (so Docker's volume-init-from-image rule populates a named volume on first attach).
2. Contain `/nix/store/<hash>-*/` — the content-addressed Nix dependencies (native binaries + the closure's Python wheel contents).
3. Contain `/nix/entry/python/<package-tree>/` — the closure's Python package, drop-importable when the runtime adds `entry/python` to `sys.path`.
4. Contain `/nix/entry/manifest.json` — a `ClosureManifest` JSON file with `abi == AGENTIX_CLOSURE_ABI` and `package = "agentix_closures.<name>"`.
5. Optionally contain `/nix/entry/bin/...` — native binaries the impl shells out to (claude, git, …). Exposed to `/exec` via `paths_from=["agentix_closures.<name>"]`.

There is no `bin/start`. The runtime is a single process; closures contribute Python modules to it, not separate binaries.

## Python package layout

The closure's Python package must declare three things:

```
agentix_closures/
└── <name>/
    ├── __init__.py        # caller-facing stubs
    ├── _impl.py           # real implementation
    └── _register.py       # def register() -> Dispatcher
```

### `__init__.py` — typed stubs

```python
from dataclasses import dataclass

@dataclass
class RunResult:
    exit_code: int
    patch: str

def run(instruction: str, workdir: str = "/testbed") -> RunResult:
    """Run against an instruction; returns a fake patch echoing the input."""
    raise NotImplementedError("call via RuntimeClient.remote(my_closure.run, ...)")
```

Signature is the contract. Body raises so a caller who accidentally invokes `run(...)` locally fails fast.

### `_impl.py` — real bodies

```python
from . import RunResult

def run(instruction: str, workdir: str = "/testbed") -> RunResult:
    # actually do work
    return RunResult(exit_code=0, patch="...")
```

Plain functions; no decorators, no FastAPI, no socket binding. May be `async def`.

### `_register.py` — bind stubs to impls

```python
from agentix.dispatch import Dispatcher
from . import run
from ._impl import run as _run

def register() -> Dispatcher:
    d = Dispatcher()
    d.bind(run, _run)
    return d
```

Runtime calls `register()` once on startup. Pure function, no globals.

## Manifest

`/nix/entry/manifest.json` is the marker that identifies a `/mnt/<dir>` mount as an Agentix closure. The runtime ignores any mount whose manifest is missing, malformed, or carries an incompatible abi.

```json
{
  "abi": 1,
  "name": "my-closure",
  "version": "1.0.0",
  "package": "agentix_closures.my_closure",
  "kind": "agent",
  "description": "Short blurb"
}
```

| Field | Required | Purpose |
|---|---|---|
| `abi` | yes | Must equal `AGENTIX_CLOSURE_ABI` (currently `1`). Runtime skips mismatches with a warning. |
| `name` | yes | Human-readable name (for logs). |
| `version` | yes | Semantic version. |
| `package` | yes | Python import path (`agentix_closures.<name>`). **Routing key.** |
| `description` | no | Short description. |
| `kind` | no | Free-form tag for tooling; runtime ignores. |

Extra fields are allowed and preserved.

## Sandbox-side placement

After the deployment puts each closure's content into a per-image named volume and mounts each at `/mnt/c<digest>:ro`, a sandbox sees:

```
/mnt/c<digest>/
├── store/<hash>-*/                         ← Nix deps (used by the symlink forest)
└── entry/
    ├── python/
    │   └── agentix_closures/<name>/        ← runtime imports this
    │       ├── __init__.py
    │       ├── _impl.py
    │       └── _register.py
    ├── bin/<cli>                           ← optional native binaries
    └── manifest.json                       ← ClosureManifest
```

and

```
/nix/
└── store/<hash>-*/                         ← tmpfs, symlinked from /mnt/*/store/*
```

Every Nix binary's absolute `/nix/store/<hash>` reference resolves through the symlink forest. Mount dir names (`/mnt/c<digest>`) are deployment-internal; the runtime indexes closures by `manifest.package`, not by directory.

## Runtime lifecycle

```
Sandbox boot
    │
    ├─ tmpfs /nix
    ├─ mkdir /nix/store
    ├─ ln -sfn /mnt/*/store/*  /nix/store/
    └─ exec /mnt/runtime/entry/bin/start
           │
           └─ lifespan: scan /mnt/*  (skip 'runtime')
                for each /mnt/<dir>/entry/manifest.json (valid + matching abi):
                    sys.path.insert(0, /mnt/<dir>/entry/python)
                    pkg = importlib.import_module(manifest.package)
                    dispatcher = pkg._register.register()
                    registry[manifest.package] = dispatcher
```

Closures are **fixed at sandbox create time**; change the set by recreating the sandbox.

## Wire

A single endpoint serves all remote calls.

```
POST /_remote
  { "package": "agentix_closures.my_closure",
    "method":  "run",
    "args":    [],
    "kwargs":  { "instruction": "fix the bug" } }
```

Success:

```json
{ "ok": true, "value": { "exit_code": 0, "patch": "..." } }
```

Failure (validation, impl raises, serialization):

```json
{ "ok": false, "error": { "type": "ValueError", "message": "...", "traceback": "..." } }
```

The wire stays 200 even for impl failures — `error.type` carries the exception class name. Only "package not loaded" returns 404.

W3C `traceparent` / `tracestate` headers pass through.

## Caller side

```python
from agentix import RuntimeClient
from agentix_closures import my_closure

async with RuntimeClient(sandbox.runtime_url) as c:
    result = await c.remote(my_closure.run, instruction="...")
    # result is my_closure.RunResult — IDE / mypy fully informed
```

`RuntimeClient.remote(fn, *args, **kwargs)`:
- Reads `fn.__module__` → routing key
- Reads `fn.__name__` → method
- Serialises args/kwargs via pydantic `TypeAdapter` from `inspect.signature(fn)`
- Decodes the response value into `fn`'s return type

No magic registration, no decorators on the stubs. The function reference and its signature carry everything the wire needs.

## Runtime built-ins

Independent of any closure, the runtime exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `POST /exec` | Run a shell command in the sandbox. Body `{command, cwd?, env?, timeout?, paths_from?}`. SSE when `Accept: text/event-stream`; else JSON `{exit_code, stdout, stderr}`. |
| `POST /upload` | Multipart upload into `AGENTIX_UPLOAD_ROOT` (default `/workspace`). |
| `GET /download?path=…` | Stream a file back. |
| `GET /closures` | List loaded closures and their manifests. |

`RuntimeClient.run / upload / download / closures` are typed Python helpers. Directory listing and other file inspection go through `/exec` (`ls -la`, `find`, `stat`).

### `/exec` env and PATH

Subprocesses run with a scrubbed env:

- Stripped: `LD_LIBRARY_PATH`, `LD_PRELOAD`, `PYTHONPATH`, `PYTHONHOME`, `LOCALE_ARCHIVE`, `FONTCONFIG_*`, `SSL_CERT_FILE`, anything prefixed `NIX_`.
- Default PATH: the task image's (`/usr/local/bin:/usr/bin:/bin`). Task-image tools take precedence over closure-bundled tools of the same name.
- Opt-in to a closure's bins with `paths_from=["agentix_closures.<name>"]` — prepends that closure's `entry/bin`. `["*"]` includes every loaded closure.

## Writing a closure (minimal recipe)

```
my-closure/
├── pyproject.toml          # [project] name = "agentix-closure-my-closure"; packages = ["agentix_closures/my_closure"]
├── agentix_closures/
│   └── my_closure/
│       ├── __init__.py     # stub: typed signatures, body raises NotImplementedError
│       ├── _impl.py        # real bodies
│       └── _register.py    # def register() -> Dispatcher: ...
├── manifest.json           # { "abi": 1, "package": "agentix_closures.my_closure", ... }
├── default.nix             # buildPythonPackage + symlinkJoin into /entry/python + /entry/manifest.json
└── Dockerfile              # nix-build → copy /export into final image
```

Build the image with a Dockerfile that runs `nix-build` in a builder stage and copies `/export` into a `VOLUME /nix` final layer. `tests/closure-docker/Dockerfile` in this repo is a working reference, as are `tests/closures/mock-agent/` and `tests/closures/mock-dataset/`.

Use it:

```python
SandboxConfig(
    image="ubuntu:24.04",
    runtime="agentix/runtime:0.1.0",
    closures=["my-closure:1.0"],
)
```
