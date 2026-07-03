# Agentix Architecture

Agentix is a framework for **agent evaluation**, **RL rollout
execution**, and **training-data collection**. Host-side trainers and
eval scripts orchestrate sandboxes; sandbox-side code is ordinary Python
(agents, bash, scorers). [`abridge`](plugins/abridge/README.md) correlates
traces into rollout logs for RL buffers. The design goal is a lower
integration tax than bespoke rollout servers — see the README comparison
with [ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server).

## The two pieces

Everything reduces to two operations, and the split between them is the
whole mental model:

1. **Bundle** — `agentix build [path]` packages one Python project (the
   framework, your code, integration modules, dependencies, optional
   system binaries) into one deploy-ready runtime image. *The bundle
   decides what code and dependencies exist in the sandbox.*
2. **Remote call** — `client.remote(fn, ...)` runs a Python callable
   inside that image from host-side Python and returns its value. *The
   remote call decides which callable runs.*

```text
Bundle              = what code and dependencies exist in the sandbox
client.remote(fn)   = which importable function to call
Worker              = where user code executes
agentix.sio         = host ↔ sandbox side channels (trace, log, plugins)
SandboxProvider     = where the bundle image runs
```

## Programming model

Pass a normal Python callable. The provider hands you a `Sandbox` with a
`remote(...)` method; `RuntimeClient` is the lower-level handle it wraps.

```python
from app import run

async with provider.session(config) as sandbox:
    result = await sandbox.remote(run, input="hello")
```

Importing the module first gives Agentix the same callable object:

```python
import app

result = await sandbox.remote(app.run, input="hello")
```

The host encodes the callable as an import-path `RemoteCallable`
(`module::qualname`). Lambdas, bound methods, partials, and other
non-importable callables are rejected at the host before the call leaves.

## Bundle

`agentix build [path]` takes one Python project and produces a
deploy-ready image.

```text
my-project/
├── pyproject.toml
├── src/app.py
└── default.nix              # optional, for system binaries
```

Python dependencies come from the project's `pyproject.toml` — installing
the project pulls in everything the sandbox needs:

```toml
[project]
name = "my-project"
version = "0.1.0"
dependencies = [
    "agentixx>=0.1.0",
    "agentix-runtime-basic>=0.1.0",   # agentix.bash, file ops
    "agentix-dataset-swe>=0.1.0",     # agentix.plugins.datasets.swe
]
```

The build splits along one hard line — **uv owns Python, Nix owns system
binaries; there is no uv2nix.** Inside the build container Agentix creates
the runtime venv and installs the full (non-editable) dependency closure
with uv:

```bash
uv venv /nix/runtime/venv
uv sync          # the project + direct + transitive deps + integration modules
```

If the project ships a `default.nix`, a Nix builder stage materializes its
derivation closure and symlinks `bin/*` into `/nix/runtime/bin`. The
result is one merged tree, mounted at `/nix`:

```text
/nix/runtime/
├── bootstrap.sh           # container entry point (provider backends exec this)
├── bin/                   # symlinkJoin of every Nix closure (e.g. git, rg)
└── venv/
    └── lib/python3.11/site-packages/
        ├── agentix/
        ├── agentix/bash/
        ├── agentix/plugins/datasets/swe/
        └── app.py
```

Worker processes inherit the runtime-server environment, with the bundle
venv and Nix bins prepended to `PATH`:

```text
/nix/runtime/venv/bin:/nix/runtime/bin:${PATH}
```

So sandbox code can call tools by name:

```python
await asyncio.create_subprocess_exec("git", "status")
await asyncio.create_subprocess_exec("claude", "-p", instruction)
```

## Remote calls

`sandbox.remote(fn, ...)` runs one callable in the sandbox and returns its
value. The host:

1. builds a `RemoteCallable` from `fn.__module__` and `fn.__qualname__`
2. pickles `(args, kwargs)` with stdlib pickle
3. sends both over Socket.IO on the `/` namespace

```python
from agentix.plugins.datasets import swe

score = await sandbox.remote(swe.score, instance=inst, patch=patch)
```

becomes a wire payload like:

```python
{
    "call_id": "…uuid…",
    "callable": "agentix.plugins.datasets.swe::score",
    "arguments": pickle.dumps(((), {"instance": inst, "patch": patch})),
}
```

Sync and async functions both work as targets; the worker awaits when the
return value is awaitable. Payloads round-trip as pickle blobs, but the two
directions are asymmetric: host→sandbox `arguments` are decoded with plain
`pickle.loads` (the host is trusted), while the sandbox→host return value —
which a less-trusted workload can shape — is decoded through a restricted
allowlist loader (`agentix.runtime.shared.safepickle`) that admits only
reviewed value types and raises the public `agentix.RestrictedUnpickleError`
on anything else. The runtime still does not run pydantic validation on the
wire.

## Flow

```text
Host
  sandbox.remote(fn, ...)
    RemoteCallable._resolve(fn)  ->  module::qualname
    pickle.dumps((args, kwargs))
      |
      v  Socket.IO `/` — call / call:result / call:error / cancel
Sandbox
  /nix/runtime/bootstrap.sh -> uvicorn -> agentix.runtime.server.app:app
      |
      v  length-prefixed msgpack frames on a private pipe
Single runtime worker process
  RemoteCallable.resolve()  ->  import fn
  pickle.loads(arguments)
  call fn(*args, **kwargs)   (awaiting when needed)
  pickle.dumps(result)
      |
      v  call:result — host decodes via safepickle.restricted_loads
        (allowlist; refuses unknown globals with RestrictedUnpickleError)
```

The msgpack codec that frames every event and side-channel payload also
validates its own extension types on decode, raising `ExtDecodeError`
(`agentix.runtime.shared.codec`) on a malformed frame rather than letting a
raw numpy/msgpack error escape.

Side channels share the same Socket.IO connection:

- `/trace` — span lifecycle from sandbox to host
- `/log` — stdlib logging records plus captured stdout/stderr
  (`agentix.sandbox.stdout` / `agentix.sandbox.stderr`) from sandbox to host
- `/<plugin>` — plugin namespaces registered via `agentix.sio`

## Worker model

The runtime server owns **one** worker subprocess that handles all remote
calls. The worker uses the same `/nix/runtime` venv as the server, so
anything installed into the bundle can be imported. For each call it:

1. resolves the `RemoteCallable` import path
2. unpickles `(args, kwargs)`
3. calls the callable (awaiting when needed)
4. pickles the return value (the host decodes it through the restricted
   allowlist loader, not plain `pickle.loads`)

The worker also captures the process's stdout and stderr — `print()`,
subprocess output, C-extension writes — replays each line on `/log` as
`agentix.sandbox.stdout` / `agentix.sandbox.stderr`, and appends it to a
durable `$AGENTIX_LOG_DIR/sandbox-<worker-id>.log` (default `/tmp/agentix`)
so output survives a dropped connection.

The single-worker model is intentional for now — it keeps runtime state
and debugging simple while the public API settles. It is an
implementation detail: future runtimes may use worker pools or per-call
isolation without changing `sandbox.remote(...)`.

## End-to-end example

```python
from agentix.bash import run as bash_run
from agentix.plugins.datasets import swe
from my_project.tasks import generate_patch

async with provider.session(config) as sandbox:
    await sandbox.remote(bash_run, command="git clone ...")
    patch = await sandbox.remote(generate_patch, prompt="fix the bug")
    score = await sandbox.remote(swe.score, patch=patch)
```

All three calls run inside the same bundle image. They target different
modules, but those modules all come from the same installed runtime
environment.
