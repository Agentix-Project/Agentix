# Project conventions

## 组合优于继承 / Composition over inheritance

**Read this three times. Say it out loud once.**

1. **组合优于继承.** This framework chooses composition over inheritance, everywhere it has the choice. Don't introduce inheritance to share behaviour, to mark relationships, or to give pyright a typing hook. Compose instead — pass an instance, register a callback, declare a Protocol.
2. **组合优于继承.** The closure ABI is the canonical example. A closure's stub class (`class Bash(Namespace)`) and its impl class (`class BashImpl`) are **independent classes that share no inheritance edge**. `_register.py` composes them by handing both to `Dispatcher.bind_namespace(Bash, BashImpl())`. `BashImpl` provides the `Bash` interface; it isn't a kind of `Bash`.
3. **组合优于继承.** When you reach for a base class to "share code" or "enforce a contract", stop. Ask: would a free function, a protocol, a wire-pattern strategy, or a deployment configuration object work instead? It almost always does. The cost of inheritance is that the parent and child are forever co-evolving; composition lets each piece change independently.

The reverse — using inheritance — is allowed only when the relationship is genuinely is-a and there's no composition alternative (e.g. `DockerDeployment` implements `Deployment`'s abstract methods because backends must satisfy a fixed lifecycle interface). Even then, prefer the smallest possible inheritance footprint.

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
- `/nix/entry/manifest.json` — `ClosureManifest` JSON with `abi == AGENTIX_CLOSURE_ABI` and `package = "agentix_closures.<name>"`. **Generated at build time** from the closure's `__init__.py` metadata (`__version__` + module docstring) by `tools/gen_manifest.py`; closure authors don't write this file.
- Optional: `/nix/entry/bin/...` — native binaries the closure's impl shells out to (claude, git, …). `/exec paths_from=[<package>]` exposes them on PATH.

### Closure source layout

A closure's source directory is the minimum-viable layout — three files:

```
primitives/<name>/
├── pyproject.toml                # all metadata: name, version, description
└── agentix_closures/<name>/
    ├── __init__.py               # stub class: `class Foo(Namespace)`
    └── _impl.py                  # impl class: `class FooImpl`
```

- **`pyproject.toml`** is the single source of metadata truth. `name` follows the `agentix-<kind>-<short>` convention (e.g. `agentix-primitive-bash`); `version` and `description` flow into the generated manifest. Nothing else needs to know the name or version.
- **`__init__.py`** is what callers import. Stub methods have `...` bodies — the signature is the contract; there is no body to run on the caller side. No metadata (`__version__` / `__image__`) goes here; the framework derives both from `pyproject.toml` via `importlib.metadata`.
- **`_impl.py`** has the real bodies on an independent class. Composition over inheritance: `FooImpl` does NOT subclass `Foo`.

Optional escape hatches:

- **`_register.py`** — declare imperative binding when the convention (one `Namespace` subclass paired with `<Name>Impl`) isn't enough. Rare.
- **`manifest.json`** — ship a pre-built manifest only if your closure intentionally diverges from what `pyproject.toml` would produce.

Build infrastructure is shared, not per-closure:

- `primitives/_template/Dockerfile` — same for every closure
- `primitives/_template/default.nix` — same for every closure; pulls metadata from the closure's `pyproject.toml`
- `tools/gen_manifest.py` — stdlib-only script that derives `manifest.json` from `pyproject.toml`; copied into the build context by `agentix build`

The runtime imports each closure lazily on first call. No global mutable state in the closure.

### CLI

Developer commands ship as the `agentix` console script (`pip install -e .[dev]` registers it):

```
agentix build primitives/bash               # build the closure image
agentix build primitives/bash --dry-run     # stage the build context to ./build/<name>/ for inspection
agentix check primitives/                   # stub ↔ impl signature drift across all closures under primitives/
```

`agentix build` stages a self-contained docker build context (closure source + shared Dockerfile/nix/gen_manifest) into a temp dir, then runs `docker build`. The closure dir never carries Dockerfile/default.nix/manifest.json itself.

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

### Runtime startup (lazy)

On lifespan startup, the runtime:

1. Scans `/mnt/*` for `entry/manifest.json`. Skips `/mnt/runtime`.
2. For each valid manifest (matching abi), prepends `<mount>/entry/python` to `sys.path` and **registers a pending entry** in the global `Registry`. **No imports run.**
3. The closure's Python package is imported and its `_register.register()` is called on **first `/_remote` request** for that package (`Registry.get_or_load`), under a per-package async lock so concurrent first-calls share one import.
4. Import failures are cached on the entry; every subsequent call returns the same error without retrying.

Two images shipping the same `package` collide — second is skipped with a warning. There are **no caller-chosen namespaces**; the Python import path is the identity.

This means: a broken closure does not block sandbox boot; an unused closure costs nothing to mount; first-call latency for a closure includes its one-time import cost (typically tens of ms).

### Wire

Two transports, used per call shape:

**Unary** — `POST /_remote` (HTTP, JSON):

```
POST /_remote
  { "package": "agentix_closures.claude_code",
    "method":  "run",
    "args":    [],
    "kwargs":  { "instruction": "fix the bug" } }

← { "ok": true, "value": { "exit_code": 0, "stdout": "...", "patch": "..." } }
```

Failures come back as `{ "ok": false, "error": {...} }`. Wire stays 200.

**Server-streaming, bidirectional, and log subscription** — Socket.IO at `/socket.io/`. One persistent Socket.IO connection per `RuntimeClient` multiplexes all such calls, demultiplexed by a caller-generated `call_id`. Event shapes:

```
stream            {call_id, package, method, args, kwargs}
stream:item       {call_id, value}
stream:end        {call_id}
stream:error      {call_id, error}

bidi:start        {call_id, package, method, args, kwargs}
bidi:in           {call_id, item}
bidi:end_in       {call_id}
bidi:out          {call_id, value}
bidi:end          {call_id}
bidi:error        {call_id, error}

logs:subscribe    {filter?: <logger-name prefix>}
log               {level, name, message, timestamp}
logs:unsubscribe  {}
```

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

### PATH policy for the `bash` primitive

Shell exec is the `bash` primitive closure (`primitives/bash/`), not a runtime built-in. Invoke via `c.remote(bash.run, command=...)`.

User subprocess default `PATH=/usr/local/bin:/usr/bin:/bin` (task image's). Nix env vars (`LD_LIBRARY_PATH`, `NIX_*`, `PYTHONPATH`, etc.) scrubbed to avoid ABI clash. `paths_from=["agentix_closures.<name>"]` prepends that closure's `entry/bin` to PATH.

### What Nix buys us

- Content-addressed `/nix/store` paths → multiple closures' deps never collide, so the symlink forest is trivially safe.
- Hermetic native binaries per closure (claude, git, …) referenced via Nix-absolute shebangs + RPATH.

### Deliberate non-choices

- **No subprocess-per-closure.** All closure impls run in the runtime's Python event loop.
- **No reverse proxy.** `POST /_remote` is direct dispatch; closures expose Python functions, not arbitrary HTTP routes.
- **No caller-chosen namespaces.** `manifest.package` is the identity. Two images shipping the same package collide.
- **Streaming returns** via `AsyncIterator[T]` annotation on the stub: `async for x in c.remote(stream_fn, ...)`. Wire is Socket.IO `stream`/`stream:item`/`stream:end` events. Bidi (stub takes one `AsyncIterator[T]` parameter and returns `AsyncIterator[U]`) is supported via the `bidi:*` event family.
- **No monolithic single-image runtime.** Each closure is its own image; the runtime image only ships `agentix` + `pydantic` + `fastapi` + `uvicorn`.

## Implementation notes

- **Hash paths are internal.** Users pass docker image refs in `SandboxConfig.closures` — either as strings or as the closure's imported Python package (which exposes `__image__` for resolution). Mount-dir names are deployment-internal (`/mnt/c<digest>`); the runtime indexes by `manifest.package`.
- **No local Nix required.** Closure authors do `docker build`; Nix lives in the builder stage of their Dockerfile.
- **Closure Python deps stay thin.** Closures share the runtime's Python interpreter — Python wrappers should depend on stdlib + the `agentix` package itself (which already brings pydantic). Heavy deps belong in Nix-bundled native binaries, not in `pyproject.toml`.
- **Sandbox starts fast.** Warm sandbox is `-v` mounts + tmpfs + symlink loop (shell-time, ~100 ms) + import of each closure package (typically tens of ms each).
- **Populate is lock-serialised** in-process to avoid concurrent `docker run -v` races on the same image's volume. Cross-process coordination is not currently provided; documented as a single-orchestrator assumption.

## Typing conventions

The wire layer is loosely typed at the protocol level (strings, JSON), so we lean on the Python type system to keep the surrounding code honest. Four house rules:

### 1. Namespace stubs + composition impls (R1)

A closure's typed surface is a `Namespace` subclass with `...`-bodied methods. The matching impl is a **separate, independent class** whose methods structurally match the stub. `_register.register()` composes them:

```python
# __init__.py
from agentix.namespace import Namespace

class Bash(Namespace):
    async def run(self, command: str) -> BashResult: ...
    async def run_stream(self, command: str) -> AsyncIterator[BashEvent]: ...

# _impl.py — no inheritance from Bash
class BashImpl:
    async def run(self, command: str) -> BashResult: ...
    async def run_stream(self, command: str) -> AsyncIterator[BashEvent]: ...

# _register.py
def register() -> Dispatcher:
    return Dispatcher.bind_namespace(Bash, BashImpl())
```

`Dispatcher.bind_namespace` walks the stub class via `agentix.namespace.discover_methods`, looks up the matching attribute on the impl instance, and calls `bind()` for each pair. Composition, not inheritance — re-read the rule three times above if tempted otherwise.

**Static type checking** is opt-in. `Namespace` is a `Protocol` so users who want pyright to verify the impl can declare:

```python
@runtime_checkable
class Bash(Namespace, Protocol):
    async def run(self, command: str) -> BashResult: ...

impl: Bash = BashImpl()  # pyright catches structural mismatch here
```

### 2. Pluggable wire patterns (R2)

Call shapes (unary / server-stream / bidi / …) live in `agentix.wire` as `WirePattern` subclasses. Each pattern owns:

* `matches(sig) -> bool` — does this signature use this pattern?
* `bind(sig)` — per-method state precompute at `Dispatcher.bind` time.

Built-ins ship as `UnaryPattern`, `StreamPattern`, `BidiPattern` and are registered in specific-to-general order. Third parties extend the framework by registering their own:

```python
from agentix.wire import WirePattern, register_pattern

class PubSubPattern(WirePattern):
    name = "pubsub"

    @classmethod
    def matches(cls, sig): ...

    def bind(self, sig): ...

register_pattern(PubSubPattern)
```

`register_pattern` prepends to the list — user patterns outrank built-ins. The Dispatcher picks the pattern at bind time and caches it on the bound method.

### 3. Branded identifiers from `agentix.idents`

There are four `str`s in the wire layer that are easy to confuse — a closure's import path, a method name, the rollout correlation key, and the sandbox handle. They are `NewType`d in `agentix/idents.py` (`PackageName`, `MethodName`, `CallId`, `SandboxId`) and consumed everywhere the wire types appear:

- `ClosureManifest.package: PackageName`
- `RemoteRequest.{package, method, call_id}`
- `TraceEvent.{call_id, source}` (source is also a `PackageName`)
- `Sandbox.sandbox_id` / `SandboxInfo.sandbox_id` / `DockerDeployment._ports`
- `Dispatcher._methods` keyed by `MethodName`, `Registry._entries` by `PackageName`
- `trace.set_call_context` / `trace.emit` / contextvars

When you write new wire-adjacent code, use the branded types — pyright treats them as distinct, so swapping `MethodName` for `PackageName` becomes a type error. Pydantic v2 understands `NewType`, so JSON round-trip is unchanged.

### 4. Stub ↔ impl signature drift is a CI failure

`tools/check_stub_impl.py` loads each closure's `_register.register()` and compares the stub's signature against the impl's for every bound method — parameter names, kinds, defaults, annotations, return type. Run it locally:

```
python tools/check_stub_impl.py            # defaults to primitives/
python tools/check_stub_impl.py path/to/closure
```

Drift causes a non-zero exit. This is the one class of bug the runtime itself cannot catch until the first call lands, so it gets caught in CI instead.

The checker is shape-agnostic: it works for both legacy module-function stubs and for the upcoming class-based `Namespace` shape, because both bottom out at `Dispatcher.bind()`.
