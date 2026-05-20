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

Both forms give Agentix the same callable object. Agentix transports
callables with stdlib pickle, so module-level functions/classes and
pickleable callable objects are the supported boundary.

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
symlinks `bin/*` into `/nix/runtime/bin/`.

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

`RuntimeClient.remote(fn, ...)` serializes the callable with stdlib
pickle and sends that callable payload plus args and kwargs to the
runtime. Pickle is Python's native callable reference mechanism.

For example:

```python
from agentix.swebench import run

score = await client.remote(run, instance=inst, patch=patch)
```

is sent as a callable payload with a display name like:

```text
agentix.swebench::run
```

The request body contains the callable payload plus serialized args and
kwargs:

```python
{
    "callable_payload": b"...pickle...",
    "display_name": "agentix.swebench::run",
    "shape": "unary",
    "args": [],
    "kwargs": {"instance": inst, "patch": patch},
}
```

The runtime worker unpickles the callable inside the sandbox and invokes
it. For importable callables this resolves to the sandbox-installed
module implementation.

Arguments are passed as msgpack payloads. Before sending, the client
uses the local function signature and type annotations to serialize
positional and keyword arguments. Inside the worker, the signature is
resolved from the unpickled callable, and pydantic validates/coerces the
received values before the callable is invoked. Return values and stream
items are serialized the same way on the way back.

## Flow

```text
Host
  RuntimeClient.remote(fn, ...)
    pickle callable
    detect unary / stream / bidi
    encode args and kwargs
      |
      v
Sandbox
  agentix-server
      |
      v
Single runtime worker process
  unpickle callable
  call fn(*args, **kwargs)
```

## Call Shapes

Agentix supports three call shapes:

| Function shape | Transport | Client usage |
| --- | --- | --- |
| normal async function | Socket.IO unary event | `await client.remote(fn, ...)` |
| async generator | Socket.IO stream | `async for item in client.remote(fn, ...)` |
| async generator with `Channel[T]` input | Socket.IO bidi | send through `Channel`, receive with `async for` |

Shape detection is structural:

- async generator -> stream
- async generator with a `Channel[T]` parameter -> bidi
- everything else -> unary

## Worker Model

The current runtime server owns one worker subprocess. That worker
handles all remote calls for the runtime. This is an implementation
detail: future runtimes may use worker pools or per-call isolation
without changing `RuntimeClient.remote(...)`.

For each call, the worker:

1. unpickles the callable payload
2. detects and verifies the expected call shape
3. validates args with pydantic
4. calls the callable
5. serializes the return value or stream items

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
```
