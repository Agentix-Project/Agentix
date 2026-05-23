# Agentix Architecture

Agentix has two core pieces:

1. **Bundle**: build one runtime image containing the framework, user
   code, integration modules, Python dependencies, and optional system
   binaries.
2. **Remote calls**: call Python callables inside that runtime image from
   host-side Python with `RuntimeClient.remote(fn, ...)`.

The important split is simple:

- Bundle decides what code and dependencies exist in the sandbox.
- `client.remote(fn, ...)` decides which callable to run.

## Programming Model

Users pass a normal Python callable:

```python
from agentix import RuntimeClient
from app import run

async with RuntimeClient(sandbox.runtime_url) as client:
    result = await client.remote(run, input="hello")
```

This form is the primary API. Importing the module first also works:

```python
import app

result = await client.remote(app.run, input="hello")
```

Both forms give Agentix the same callable object. The host encodes it as
an import-path `RemoteCallable` (`module::qualname`). Lambdas, bound
methods, partials, and other non-importable callables are rejected at
the host before the call leaves.

## Bundle

`agentix build [path]` takes one Python project and produces a
deploy-ready image.

```text
my-project/
├── pyproject.toml
├── src/app.py
└── default.nix              # optional, for system binaries
```

Python dependencies come from the project's `pyproject.toml`:

```toml
[project]
name = "my-project"
version = "0.1.0"
dependencies = [
    "agentixx>=0.1.0",
    "agentix-runtime-basic>=0.1.0",
    "agentix-swebench>=0.1.0",
]
```

During build, Agentix stages the source and runs one install into the
runtime venv:

```bash
/nix/runtime/bin/pip install --no-cache-dir /src/project
```

That single install brings in:

- the user project
- direct dependencies
- transitive dependencies
- integration modules such as `agentix.bash` or `agentix.swebench`

At runtime, all installed modules live in the same Python environment:

```text
/nix/runtime/
├── bin/
│   ├── python
│   ├── pip
│   └── agentix-server
└── lib/python3.11/site-packages/
    ├── agentix/
    ├── agentix/bash/
    ├── agentix/swebench/
    └── app.py
```

If the project includes `default.nix`, `agentix build` adds a Nix
builder stage, copies the derivation closure into the final image, and
symlinks `bin/*` into `/nix/runtime/bin`.

Worker processes inherit the runtime server environment, with the
bundle venv and Nix runtime bins prepended to `PATH`:

```text
/nix/runtime/venv/bin:/nix/runtime/bin:${PATH}
```

So sandbox code can call tools by name:

```python
await asyncio.create_subprocess_exec("git", "status")
await asyncio.create_subprocess_exec("claude", "-p", instruction)
```

## Remote Calls

`RuntimeClient.remote(fn, ...)` runs one Python callable in the sandbox
and returns its value. The host:

1. builds a `RemoteCallable` from `fn.__module__` and `fn.__qualname__`
2. pickles `(args, kwargs)` with stdlib pickle
3. sends both over Socket.IO on the `/` namespace

For example:

```python
from agentix.swebench import run

score = await client.remote(run, instance=inst, patch=patch)
```

becomes a wire payload like:

```python
{
    "call_id": "…uuid…",
    "callable": "agentix.swebench::run",
    "arguments": pickle.dumps(((), {"instance": inst, "patch": patch})),
}
```

The runtime server forwards the request to one worker subprocess. The
worker imports the module, resolves the function, unpickles args/kwargs,
calls it (awaiting when the return value is awaitable), and pickles the
result back.

Sync and async functions both work as targets. Args and return values
round-trip as pickle blobs; the runtime does not run pydantic
validation on the wire today.

## Flow

```text
Host
  RuntimeClient.remote(fn, ...)
    RemoteCallable._resolve(fn)  ->  module::qualname
    pickle.dumps((args, kwargs))
      |
      v  Socket.IO `/` — call / call:result / call:error / cancel
Sandbox
  agentix-server
      |
      v  length-prefixed msgpack frames on a private pipe
Single runtime worker process
  RemoteCallable.resolve()  ->  import fn
  pickle.loads(arguments)
  call fn(*args, **kwargs)
  pickle.dumps(result)
```

Side channels on the same Socket.IO connection:

- `/trace` — span lifecycle from sandbox to host
- `/log` — stdlib logging records from sandbox to host
- `/<plugin>` — plugin namespaces registered via `agentix.sio`

## Worker Model

The current runtime server owns one worker subprocess. That worker
handles all remote calls for the runtime. This is an implementation
detail: future runtimes may use worker pools or per-call isolation
without changing `RuntimeClient.remote(...)`.

For each call, the worker:

1. resolves the `RemoteCallable` import path
2. unpickles `(args, kwargs)`
3. calls the callable (awaiting when needed)
4. pickles the return value

The worker uses the same `/nix/runtime` venv as the runtime server, so
anything installed into the bundle can be imported by the worker.

## End-to-End Example

```python
from agentix import RuntimeClient
from agentix.bash import run as bash_run
from agentix.swebench import run as score_swebench
from my_project.tasks import generate_patch

async with RuntimeClient(sandbox.runtime_url) as client:
    await client.remote(bash_run, command="git clone ...")
    patch = await client.remote(generate_patch, prompt="fix the bug")
    score = await client.remote(score_swebench, patch=patch)
```

All three calls run inside the same bundle image. They may target
different modules, but those modules all come from the same installed
runtime environment.

## Mental Model

```text
Bundle = what code and dependencies exist in the sandbox
client.remote(fn) = which importable function to call
Worker = where user code executes
agentix.sio = host ↔ sandbox side channels (trace, log, plugins)
Deployment = where the bundle image runs
```
