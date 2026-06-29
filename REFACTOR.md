# Convergence refactor — what we're building

Design baseline. Some items done, the rest planned (see Work below).

## Principles

- **Programming framework.** Make `async def rollout(sandbox, task) -> R`
  pleasant and typed. The hero is the user's rollout function.
- **Two surfaces.** Typed code (rollout authors: `Sandbox` /
  `SandboxConfig` / `SandboxProvider` / `remote()` / plugins / `trace`)
  and a string CLI (`agentix build` / `agentix deploy` + registries).
  `deploy` is a CLI process; the bundle ref enters code as a `str`;
  providers are constructed by typed import.
- **Type what we own.** Public API, `Result[T]`, plugin contracts, and
  build-generated stubs are fully typed by construction.

## Reliability contract

The framework guarantees the integrity of *information about* each call.

- Every `remote()` resolves to `Result<T>` = **`Ok | Failed`** — a
  truthful, unique terminal state. It never hangs and runs `fn`
  at-most-once.
- **`resume`** re-fetches a terminal state (no re-run). **`retry`** is a
  new `call_id` (a new rollout); the caller owns retry.

Per channel:

| Channel | Guarantee |
|---|---|
| **RPC** | at-most-once execution; terminal state delivered exactly once; an undeliverable result is a `Failed` |
| **trace** (the dataset) | at-least-once + host-side dedup + durable sink |
| **log** | durable on disk + best-effort live stream |

## Target architecture

- **One transport:** Socket.IO `/rpc`. HTTP serves `/health`.
- **Namespaces** stay pipe-forwarded and plugin-agnostic.
- **RPC and trace** each carry their reliability; share one retain/ack
  buffer where it is genuinely cleaner.
- **log** is stdout/stderr line capture (per-sandbox file + best-effort
  stream on the `/log` namespace) — a plain forwarded namespace, no
  `ReliableStream` and no structured `LogRecord` bridge.
- **One typed plugin primitive** (below).
- **Typing is Python-native:** ParamSpec + `Result[T]` + build-time
  codegen. The build step is our compile step.

## Plugin primitive

One declarative, typed surface: define the contract once, implement one
side, call typed from the other. Default is request/response; streaming
is an explicit opt-in.

```python
# contract.py — one typed interface, shared by both sides
class Solve(BaseModel):    task: str
class Solution(BaseModel): patch: str

# host side — implement, fully typed
class Solver(Plugin):
    name = "solver"                                   # -> /solver
    @on
    async def solve(self, req: Solve) -> Solution:    # typed in/out
        return Solution(patch=await call_model(req.task))

    @stream                                           # at-least-once opt-in
    async def span(self, ev: SpanEvent) -> None: ...

# sandbox target fn — typed call
async def rollout():
    sol = await plugins.solver.solve(Solve(task="..."))
    return sol.patch
```

- One base (`Plugin`), one registration
  (`provider.session(..., plugins=[Solver()])`).
- The method *is* the op (decorator-discovered); pydantic payloads;
  framework-managed `request_id` and error envelope.
- abridge is a specialization on top (op names = HTTP paths + an
  in-sandbox FastAPI tunnel).
- Decide: the sandbox-side caller handle stays typed via a
  contract-typed handle (preferred) over a metaclass-routed one class.

## Build-time typed remote

`@remote` annotation → `agentix build` emits typed client stubs + a
closed dispatch manifest + a schema'd codec. The generated boundary is
fully typed by construction. `sandbox.remote(fn, ...)` keeps its
ParamSpec `(P) -> R`; codegen adds closed-set validation, a real schema,
and (when wanted) non-Python clients.

## Work

### Done — branch `refactor/single-transport`

1. **Single transport.** Removed the HTTP `/call` fast-path; every
   `c.remote()` rides Socket.IO `/rpc`. ruff + pyright clean; 285 tests
   pass.
2. **No silent loss.** A `resume` for an evicted/unknown `call_id`
   returns a definite `call:error` (`ResultUnavailable`); terminal
   states are `{result, error}`.
3. **Never-hang guarantee satisfied.** The worker already fails every
   in-flight call on death (`WorkerProcessExited`), fails fast on a
   closed worker, turns an oversized result frame into a `FrameTooLarge`
   error, and cancels idempotently — `remote()` always reaches a
   terminal state (or `CallTimeout`).
4. **Narrowed the top-level surface.** Moved `providers`,
   `register_provider`, `BundleDeployer`, `DeployedBundle` off
   `agentix.__all__` (still in `agentix.provider.base`); stripped the
   codec to plain msgpack; collapsed the reconnection options to
   socketio's defaults.
5. **`Result[T]` API.** `Ok | Failed` in `agentix.runtime.client.result`,
   exported at top level; `remote()` still raises, `try_remote()`
   returns `Result[R]` for exhaustive `match`.
6. **log → Ray-style capture.** The worker captures its stdout *and*
   stderr (stdlib `logging` writes to stderr, so it's captured too),
   appends to a sandbox-side `sandbox.log`, and streams each line
   best-effort on `/log`; the host replays under `agentix.sandbox.{stdout,
   stderr}`. Deleted the structured `LogRecord` bridge (`WorkerLogHandler`,
   `emit_worker_record`, the host `_replay_record`) and `/log`'s
   `ReliableStream` use. `configure_logging` stays as the local-logging
   helper (host/runtime/worker).

### To do — in order

1. **trace reshape.** Prompt per-span at-least-once emit + host durable
   sink.
2. **Shared retain/ack buffer** for RPC + trace, where it is cleaner.
3. **Plugin primitive.** Typed `Plugin` + `@on` / `@stream`; abridge as
   a specialization.
4. **Build-time typed remote bindings.** `@remote` → build emits stubs +
   manifest + schema codec.
