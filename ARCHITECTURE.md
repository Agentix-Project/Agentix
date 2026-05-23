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
a `RemoteCallable` import path (`module::qualname`). Only importable
top-level functions and builtins are supported — lambdas, partials,
bound methods, and callable instances are rejected at the host.

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

`RuntimeClient.remote(fn, ...)` is a unary RPC: one request, one pickled
return value.

For example:

```python
from agentix.swebench import run

score = await client.remote(run, instance=inst, patch=patch)
```

The wire payload looks like:

```python
{
    "call_id": "uuid-hex",
    "callable": "agentix.swebench::run",
    "arguments": pickle.dumps(((), {"instance": inst, "patch": patch})),
}
```

On the worker:

1. `RemoteCallable.resolve()` imports `agentix.swebench` and looks up `run`.
2. `pickle.loads(arguments)` yields `(args, kwargs)`.
3. The invoker calls `run(*args, **kwargs)` (awaiting if the result is a
   coroutine).
4. The return value is pickled back into `value`.

Args, kwargs, and return values travel as stdlib pickle blobs. There is
no pydantic validation on the RPC path today. Socket.IO event envelopes
use msgpack via `agentix.runtime.shared.codec`.

Sync and async functions both work as call targets.

For host↔sandbox traffic that doesn't fit `c.remote()`'s
request/response shape (trace, logs, plugin events), use
`agentix.sio` namespaces (see Side Channels below).

```text
Host
  RuntimeClient.remote(fn, ...)
    RemoteCallable._resolve(fn)  ->  module::qualname
    pickle.dumps((args, kwargs))
      |
      v  Socket.IO call / call:result / call:error
Sandbox
  agentix-server
      |
      v  length-prefixed msgpack frames
Single runtime worker subprocess
  RemoteCallable.resolve() -> fn
  pickle.loads(arguments)
  call fn(*args, **kwargs)
  pickle.dumps(result)
```

## Side Channels

Unary RPC covers request/response. Streaming and host↔sandbox events
use separate Socket.IO namespaces bridged through the worker pipe:

| Namespace | Purpose |
| --- | --- |
| `/` | unary RPC (`call`, `call:result`, `call:error`, `cancel`) |
| `/trace` | Trace/Span lifecycle (`trace_start`, `span_end`, …) |
| `/log` | stdlib `logging` records from the worker |
| `/<plugin>` | plugin-defined events (e.g. abridge LLM proxy) |

Sandbox plugins subclass `agentix.Namespace` and call
`agentix.register_namespace(...)`. Host plugins subclass
`agentix.AsyncClientNamespace` and register via
`RuntimeClient.register_namespace(...)` before connecting.

See `agentix/runtime/PROTOCOL.md` for the full wire contract.

## Worker Model

The current runtime server owns one worker subprocess. That worker
handles all remote calls for the runtime. This is an implementation
detail: future runtimes may use worker pools or per-call isolation
without changing `RuntimeClient.remote(...)`.

For each call, the worker:

1. resolves the `RemoteCallable` import path
2. unpickles `(args, kwargs)`
3. calls the callable (awaiting coroutine results)
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
client.remote(fn) = which callable to call
Worker = where the callable executes
Side-channel namespaces = host↔sandbox events beyond unary RPC
```
