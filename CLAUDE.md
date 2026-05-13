# Project conventions

## No backward compatibility

This repo is in active design. **Breaking changes are fine; do not introduce backward-compat shims.**

- **No aliases.** Rename `foo` → `bar`: delete `foo`, don't accept both.
- **No deprecation warnings.** Delete the thing.
- **No `// removed ...` / `// kept for compat` comments.** Git history covers that.
- **No version-bump fences.** Update code, docs, tests, move on.
- **Tests:** update them to the new shape; don't keep a test that exercises removed behavior.

Downstream repos (`Agentix-Agents-Hub`, `Agentix-Datasets`) are updated in lockstep — assume they follow HEAD.

## Architecture (typed Python closures, in-process dispatch)

The substrate is a single Python runtime process inside a sandbox container, into which **multiple closure images contribute Python packages**. Each closure is a typed Python module: caller imports its stubs to get full IDE / mypy support, the runtime imports the same package's `_impl` and `_register` to actually execute calls. There is no subprocess per closure, no UDS, no reverse-proxy.

### Closure image convention

Every closure image satisfies exactly:

- `VOLUME /nix` — required by the docker deployment's volume-init-from-image populate step
- `/nix/store/<hash>-*/` — content-addressed Nix deps (native binaries, libs, the closure's Python package wheel content)
- `/nix/entry/python/<package-tree>/` — the closure's Python package. The runtime adds this to `sys.path` and imports the package named in the manifest.
- `/nix/entry/manifest.json` — `ClosureManifest` JSON with `abi == AGENTIX_CLOSURE_ABI` and `package = "agentix_closures.<name>"`.
- Optional: `/nix/entry/bin/...` — native binaries the closure's impl shells out to (claude, git, …). `/exec paths_from=[<package>]` exposes them on PATH.

### Closure Python package layout

The Python package the closure ships must declare three things:

```
agentix_closures/
└── <name>/
    ├── __init__.py        # stub: typed function signatures (body: raise NotImplementedError)
    ├── _impl.py           # real implementations (only the sandbox imports this)
    └── _register.py       # def register() -> Dispatcher
```

- **`__init__.py`** is what callers import. Functions have `...` or `raise NotImplementedError` bodies — the signature is the contract; there is no body to run on the caller side.
- **`_impl.py`** has the real bodies. Plain functions; no decorators, no FastAPI, no socket binding.
- **`_register.py`** exposes `register() -> Dispatcher` that binds each stub to its impl:
  ```python
  from agentix.dispatch import Dispatcher
  from . import run
  from ._impl import run as _run

  def register() -> Dispatcher:
      d = Dispatcher()
      d.bind(run, _run)
      return d
  ```

The runtime calls `register()` once on startup. No global mutable state in the closure.

### Sandbox layout at runtime

```
/nix/                            — tmpfs (writable by entrypoint only)
  store/                         — symlink forest: each /mnt/*/store/<hash> linked here
/mnt/
  runtime/                       — runtime image's /nix slice
    store/<hash>-*/
    entry/bin/start              — agentix-server
    entry/manifest.json
  c<digest>/                     — closure image's /nix slice (dir name is internal)
    store/<hash>-*/
    entry/python/agentix_closures/<name>/...
    entry/bin/<cli>              — optional native binaries
    entry/manifest.json
```

Sandbox entrypoint (inlined into the `docker run` command):
```sh
mkdir -p /nix/store
for d in /mnt/*/store; do ln -sfn "$d"/* /nix/store/; done
exec /mnt/runtime/entry/bin/start
```

### Runtime startup

On lifespan startup, the runtime:

1. Scans `/mnt/*` for `entry/manifest.json`. Skips `/mnt/runtime`.
2. For each valid manifest (matching abi), prepends `<mount>/entry/python` to `sys.path`.
3. Imports `manifest.package` and `<package>._register`, calls `register()` to obtain a `Dispatcher`.
4. Adds the Dispatcher to a global `Registry` keyed by `manifest.package`.

Two images shipping the same `package` collide — second is skipped with a warning. There are **no caller-chosen namespaces**; the Python import path is the identity.

### Wire

A single endpoint serves all remote calls:

```
POST /_remote
  { "package": "agentix_closures.claude_code",
    "method":  "run",
    "args":    [],
    "kwargs":  { "instruction": "fix the bug" } }

← { "ok": true, "value": { "exit_code": 0, "stdout": "...", "patch": "..." } }
```

Failures (validation error, impl exception, serialization error) come back as `{ "ok": false, "error": {...} }`. Wire stays 200.

Runtime built-ins (`/exec`, `/upload`, `/download`, `/health`, `/closures`) live alongside `/_remote` at the runtime root, unrelated to closure dispatch.

### Caller side

```python
from agentix import RuntimeClient
from agentix_closures import claude_code

async with RuntimeClient(sandbox.runtime_url) as c:
    result = await c.remote(
        claude_code.run,
        instruction="fix the bug",
        workdir="/workspace",
    )
    # `result: RunResult` — IDE / mypy infer from claude_code.run's return type
```

`RuntimeClient.remote(fn, *args, **kwargs)` reads `fn.__module__` (routing key) + `fn.__name__` (method), serialises via pydantic `TypeAdapter` driven by `inspect.signature(fn)`, decodes the response into `fn`'s return type.

### PATH policy for `/exec`

User subprocess default `PATH=/usr/local/bin:/usr/bin:/bin` (task image's). Nix env vars (`LD_LIBRARY_PATH`, `NIX_*`, `PYTHONPATH`, etc.) scrubbed to avoid ABI clash. `paths_from=["agentix_closures.<name>"]` prepends that closure's `entry/bin` to PATH.

### What Nix buys us

- Content-addressed `/nix/store` paths → multiple closures' deps never collide, so the symlink forest is trivially safe.
- Hermetic native binaries per closure (claude, git, …) referenced via Nix-absolute shebangs + RPATH.

### Deliberate non-choices

- **No subprocess-per-closure.** All closure impls run in the runtime's Python event loop.
- **No reverse proxy.** `POST /_remote` is direct dispatch; closures expose Python functions, not arbitrary HTTP routes.
- **No caller-chosen namespaces.** `manifest.package` is the identity. Two images shipping the same package collide.
- **No streaming returns yet.** `RuntimeClient.remote` is request/response. Streaming wire is reserved for later.
- **No monolithic single-image runtime.** Each closure is its own image; the runtime image only ships `agentix` + `pydantic` + `fastapi` + `uvicorn`.

## Implementation notes

- **Hash paths are internal.** Users pass Docker image refs in `SandboxConfig.closures` (list of refs, not a dict). Mount-dir names are deployment-internal (`/mnt/c<digest>`); the runtime indexes by `manifest.package`.
- **No local Nix required.** Closure authors do `docker build`; Nix lives in the builder stage of their Dockerfile.
- **Closure Python deps stay thin.** Closures share the runtime's Python interpreter — Python wrappers should depend on stdlib + the `agentix` package itself (which already brings pydantic). Heavy deps belong in Nix-bundled native binaries, not in `pyproject.toml`.
- **Sandbox starts fast.** Warm sandbox is `-v` mounts + tmpfs + symlink loop (shell-time, ~100 ms) + import of each closure package (typically tens of ms each).
- **Populate is lock-serialised** in-process to avoid concurrent `docker run -v` races on the same image's volume. Cross-process coordination is not currently provided; documented as a single-orchestrator assumption.
