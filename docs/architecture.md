# Agentix Architecture (v0.1.0)

## Scope

v0.1.0 ships exactly three concerns:

1. A **closure convention** — what a Docker image must contain to be consumable by Agentix.
2. A **runtime server** — one Python process per sandbox that imports each mounted closure's Python package and exposes typed remote dispatch + sandbox I/O.
3. A **Docker deployment** — packages closures into named volumes, assembles sandboxes, starts the runtime.

See [`ROADMAP.md`](../ROADMAP.md) for what comes later.

## Components

```
┌─ Host (orchestrator) ─────────────────────────────────────────┐
│  RuntimeClient                                                 │
│    • run / upload / download           (runtime built-ins)     │
│    • closures                           (introspection)        │
│    • remote(fn, *args, **kwargs)        (typed dispatch)       │
└──────────────────────────────┬─────────────────────────────────┘
                               │ HTTP (POST /_remote)
┌─ Sandbox ──────────────────────▼───────────────────────────────┐
│                                                                 │
│  agentix-server (single Python process)                         │
│    built-in I/O:                                                │
│      GET  /health                                               │
│      POST /exec     (SSE or JSON)                               │
│      POST /upload                                               │
│      GET  /download                                             │
│      GET  /closures                                             │
│    typed dispatch:                                              │
│      POST /_remote   { package, method, args, kwargs }          │
│                                                                 │
│  Registry: package → Dispatcher (in-process, no subprocesses)   │
│    populated at startup by importing each /mnt/<dir>/entry/     │
│    python/<package>/ and calling <package>._register.register() │
│                                                                 │
│  /nix/store — tmpfs with a symlink forest merged from every     │
│  /mnt/<dir>/store content-addressed directory                   │
└─────────────────────────────────────────────────────────────────┘
```

The runtime's lifespan scans `/mnt` at startup and imports each closure's Python package. Closures are fixed for the sandbox's lifetime; change the set by recreating the sandbox.

## Closure convention

A closure is a Docker image that declares `VOLUME /nix` and carries:

- `/nix/store/<hash>-*/` — content-addressed Nix dependencies (the transitive closure)
- `/nix/entry/python/agentix_closures/<name>/` — Python package the runtime imports
  - `__init__.py` — typed stubs (caller imports)
  - `_impl.py` — real implementation
  - `_register.py` — `def register() -> Dispatcher`
- `/nix/entry/manifest.json` — `ClosureManifest` with `abi == AGENTIX_CLOSURE_ABI` and `package = "agentix_closures.<name>"`
- Optional: `/nix/entry/bin/...` — native binaries the impl shells out to

Routing is by `manifest.package`; there are no caller-chosen namespaces. Two images shipping the same `package` collide — the second mount is skipped with a warning.

See [`closure-protocol.md`](closure-protocol.md) for the full ABI.

## Sandbox layout

```
/
├── mnt/
│   ├── runtime/       ← -v agentix-closure-<digest>:/mnt/runtime:ro
│   │   ├── store/<hash>-*/
│   │   └── entry/
│   │       └── bin/start   ← the agentix-server binary
│   └── c<digest>/     ← one mount per closure, ro; dir name is internal
│       ├── store/<hash>-*/
│       └── entry/
│           ├── python/agentix_closures/<name>/
│           ├── bin/<cli>           (optional)
│           └── manifest.json
│
├── nix/
│   └── store/         ← --tmpfs /nix (writable),
│                        populated at entrypoint-time with
│                        `ln -sfn /mnt/*/store/* /nix/store/`
│
└── (task image rootfs — /usr, /bin, /etc, /testbed, ...)
```

Sandbox entrypoint (inlined into `docker run`):

```sh
set -e
mkdir -p /nix/store
for d in /mnt/*/store; do ln -sfn "$d"/* /nix/store/; done
exec /mnt/runtime/entry/bin/start
```

Why the symlink forest: Nix binaries have `/nix/store/<hash>` hard-coded in shebangs and RPATH. They only work if `/nix/store/<hash>` resolves. Symlinking each closure's `store/<hash>` into a shared `/nix/store` merges them cheaply — content-addressed paths can't collide, and the task image sees one unified `/nix/store`.

## Environment & PATH policy

Rules at every `/exec` invocation:

1. **Strip Nix-host-only env vars** — `LD_LIBRARY_PATH`, `LD_PRELOAD`, `PYTHONPATH`, `PYTHONHOME`, `LOCALE_ARCHIVE`, `FONTCONFIG_*`, `SSL_CERT_FILE`, anything prefixed `NIX_`.
2. **PATH defaults to the task image's default** (`/usr/local/bin:/usr/bin:/bin`). Task-image tools take precedence over closure-bundled tools of the same name.
3. **Opt-in closure bins** — `paths_from=["agentix_closures.<name>"]` prepends that closure's `entry/bin`. `["*"]` includes all loaded.
4. **Closure Python impls run in the runtime's interpreter** — they invoke native tools via `subprocess` with absolute `/nix/store` paths, which resolve via the symlink forest.

## Deployment (Docker)

Per unique closure image (cached in process):

```
docker run --rm -v agentix-closure-<digest>:/nix <image> true
```

Docker's volume-init-from-image rule auto-populates the named volume from the image's `/nix` layer on first attach; skips if already populated. The volume key is the image's SHA256 digest, so rebuilds produce a fresh volume automatically.

Sandbox create:

```
docker run -d \
  --name <sandbox-id> \
  --network host \
  -v agentix-closure-<runtime-digest>:/mnt/runtime:ro \
  -v agentix-closure-<digest>:/mnt/c<digest>:ro   (per closure) \
  --tmpfs /nix:exec,mode=755 \
  -e AGENTIX_BIND_PORT=<port> \
  <task-image> sh -c '<entrypoint>'
```

## Design decisions

- **In-process dispatch** — closures are Python modules in the runtime's interpreter; no subprocess, no UDS, no reverse-proxy. Cheaper, simpler, fully typed.
- **Module path = routing key** — `manifest.package` is the identity. No caller-chosen namespaces; no metadata to drift.
- **Typed stubs are the API spec** — IDE and mypy enforce parameter types; no separate schema artifact.
- **Static closure set per sandbox** — change the set by recreating the sandbox.
- **Built-in sandbox I/O on the runtime** — run / upload / download always available.
- **Closures share the runtime's Python interpreter** — Python wrappers stay thin (stdlib + pydantic, which the runtime already ships). Heavy deps belong in Nix-bundled native binaries.

## Out of scope (v0.1.0)

- Bearer-token auth on the runtime (sandbox-level trust assumed).
- Streaming returns from `remote(...)` (request/response only; reserved for v0.2).
- Higher-level interfaces for agents / datasets / benchmarks — see [`ROADMAP.md`](../ROADMAP.md).
