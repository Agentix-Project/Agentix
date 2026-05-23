# Roadmap

Agentix keeps two user-facing concepts:

- **Remote calls.** `c.remote(fn, ...)` calls a callable target inside a
  sandbox. The callable is encoded as an import-path `RemoteCallable`;
  args and kwargs travel as a pickle blob.
- **Bundle.** `agentix build [path]` packages one project root and its
  declared dependencies into a deploy-ready runtime image.

Everything below should preserve that surface. Internal worker topology,
transport choice, and deployment backend details should remain opaque to
downstream users of the library.

## v0.1.0 — RPC + Bundle

Current architecture:

- [x] `RuntimeClient.remote(fn, ...)` runs an importable callable in the
      sandbox and returns its value.
- [x] One runtime server per sandbox image.
- [x] One worker subprocess per runtime server.
- [x] Import-path `RemoteCallable` for function identity; pickle for
      args, kwargs, and return values.
- [x] Callable invocation inside `agentix.runtime.server`; targets are not
      required to be pure functions. If Python can resolve the callable
      from the requested target, Agentix should be able to invoke it.
- [x] Single-spec `agentix build`; integrations arrive through normal
      Python dependencies.
- [x] One merged `/nix/runtime` venv containing the framework, user
      project, integrations, and transitive dependencies.
- [x] Deployment backend plugin axis via `agentix.deployment`.
- [x] Side channels over the same Socket.IO connection: `/trace`, `/log`,
      and plugin namespaces via `agentix.sio`.

The single-worker model is intentional for now. It keeps runtime state
and debugging simple while the public API is still being shaped.

## Architectural Direction

### Worker Model

Keep one worker process as the default near-term runtime model.

Future improvements may add:

- worker pools
- per-call worker isolation
- concurrency limits
- CPU-bound call offloading
- restart and health policies

These changes must be opaque to downstream users. Code written as:

```python
result = await client.remote(run, input="hello")
```

should not change if the runtime later moves from one worker to many
workers.

### Callable Targets

Agentix should not require targets to be pure functions.

The runtime may call any resolved callable target, including callables
that close over module state, mutate sandbox-local state, call CLIs,
read/write files, or interact with benchmark harnesses. Purity is a user
or integration concern, not a framework constraint.

The framework's responsibility is narrower:

- encode importable callables as `RemoteCallable`
- unpickle args/kwargs and invoke the target inside the sandbox
- pickle the return value back
- surface errors in-band through the runtime protocol

Future work may add optional annotation-driven validation/coercion on
top of pickle without changing the default path.

### Transport Strategy

`c.remote()` and side channels share one Socket.IO connection. HTTP is
kept only for `/health` and the internal `/call` fast-path used by
`RuntimeClient.remote` to skip a SIO round-trip for short-running
calls.

`c.remote()` uses the `/` namespace (`call`, `call:result`,
`call:error`, `cancel`, plus `resume`/`ack` for reconnect-safe
delivery). Trace, log, and plugin traffic use dedicated namespaces
bridged through the worker pipe via `agentix.sio`; the core `/log`
and `/trace` namespaces ride `agentix.sio.ReliableStream` for
at-least-once delivery across reconnects.

Remaining transport work:

- optional annotation-driven msgpack codec path alongside pickle
- collapse event naming if the current `call:*` family becomes noisy

## Sibling Repos

Sibling repos are updated in lockstep with Agentix HEAD while the design
is still moving quickly.

- [`Agentix-Runtime-Basic`](https://github.com/Agentiix/Agentix-Runtime-Basic)
  — `bash` and `files` modules. Published as `agentix-runtime-basic`.
- [`Agentix-Deployment-Docker`](https://github.com/Agentiix/Agentix-Deployment-Docker)
  — local Docker backend. Published as `agentix-deployment-docker`.
- [`Agentix-Deployment-Daytona`](https://github.com/Agentiix/Agentix-Deployment-Daytona)
  and [`Agentix-Deployment-E2B`](https://github.com/Agentiix/Agentix-Deployment-E2B)
  — hosted deployment backends.
- [`abridge`](https://github.com/Agentiix/abridge) — host-side
  rollout-to-RL-buffer bridge.

## Later

Future directions, listed so the framework can avoid architectural
dead-ends without expanding the current API prematurely.

- **~~OpenTelemetry trace export~~ (shipped — `agentix-trace-otel`).** ship `agentix.utils.trace` spans to a
  production observability platform (Datadog, Jaeger, Tempo, Honeycomb,
  any OTLP-compatible backend). Implementation should not change the
  `agentix.utils.trace` public API. Plan:

  - Keep `agentix.utils.trace` (`Trace`, `Span`, `Processor`) as the
    user surface. Sandbox code stays unchanged.
  - The sandbox already streams `/trace` via `ReliableStream`; the host
    receives via `HostTraceNamespace` and fans out through the existing
    provider, so a new `Processor` is the right plug-in point.
  - Ship as a separate plugin package `agentix-trace-otel` to keep
    `opentelemetry-*` out of core dependencies (matches the current
    plugin-axis style of deployments / runtime-basic / agents).
  - Map `agentix.Span` → OTel `ReadableSpan`: `trace_id` / `span_id` /
    `parent_id` / `attrs` / `started_at` / `ended_at` / `status` /
    `events` are 1:1; only the id-length normalization and timestamp
    units (ns) need adapters.
  - Export from the **host** by default (sandboxes are ephemeral; host
    owns the long-lived collector connection). A sandbox-side exporter
    is possible later for cases where the sandbox can reach the
    collector directly.
  - User surface:
    ```python
    from agentix.utils import trace
    from agentix.utils.trace.otel import OTelExporter

    trace.add_processor(OTelExporter(endpoint="...", headers={...}))
    ```

- **Trace pub/sub** — remote functions emit structured rollout events;
  subscribers receive rollout-scoped fan-out.
- **RolloutPool** — warm sandbox pool for batched RL rollouts.
- **LLM proxy** — transparent proxy for API calls from remote functions,
  enabling token-level trajectory capture, cost tracking, and replay.
- **Checkpoint / partial rollout** — snapshot a sandbox filesystem and
  loaded runtime state, then fork to explore alternative continuations.
- **K8s deployment backend** — `Deployment` implementation using the
  same bundle-image contract, likely shipping as `agentix-deployment-k8s`.
