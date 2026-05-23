# `agentix.gateway` Roadmap

`agentix.gateway` is Agentix's coordinated sandbox-session service —
the same shape ProRL-Agent-Server exposes as `polar.gateway`, adapted
to Agentix's two-concept model (`c.remote(...)` + bundles).

This document tracks what's in this PR vs. what's planned. Items are
intentionally additive: nothing here should require a breaking change
to the surface introduced in PR #41 (this PR).

## Shipped today

### Core data model

* `Session`, `SessionSpec`, `SessionResult`, `SessionStatus` (`session.py`).
* `SessionStore` (live + terminal sessions, bounded), `RecordStore`
  (LLM-call records, bounded, with `for_session()`) (`storage.py`).
* `CompletionWriter` Protocol + `NullCompletionWriter`,
  `JsonlCompletionWriter` (`completion_writer.py`).

### Dispatch + stages

* `Dispatcher`: INIT -> READY -> RUNNING -> POSTRUN orchestration via
  asyncio tasks; semaphore-bounded concurrency; pause / resume gate
  between READY and RUNNING; `result_callback` for terminal handoff
  (`dispatcher.py`).
* `_resolve_callable("module::qualname")` matches
  `RuntimeClient.remote`'s import-path shape.

### HTTP surface

* `build_app(dispatcher)` exposes `/health`, `/sessions[/...]`,
  `/records`, `/pause`, `/resume` (`server.py`).
* `GatewayNode` boots the dispatcher + uvicorn + optional
  coordinator heartbeat + result-forwarding callback (`node.py`).

### Capture + transform

Capture/translate/forward for LLM traffic is the abridge plugin's
job (`agentix.bridge`). The gateway wires it in via
`GatewayNode(host_namespace_factory=lambda session:
OpenAICompatibleClient(...))`; captured records ride the existing
`/abridge` SIO namespace and the gateway funnels them into its
`RecordStore` by session id.

The earlier draft of this package shipped re-export modules
`agentix.gateway.detection` / `agentix.gateway.transform` /
`agentix.gateway.proxy`, but pyright's static analysis cannot follow
the cross-plugin `pkgutil.extend_path` namespace extension between
core `agentix.*` and plugin-shipped `agentix.bridge.*`. We dropped
the re-export wrappers; callers use the bridge's own surface
directly. The proxy-handler logic the user-facing flow needs is the
bridge's `agentix.bridge.proxy` plus the host-side
`OpenAICompatibleClient`.

### Tests

21 unit tests in `tests/gateway/` cover:

* Dispatcher: stage progression, pause-gate, error handling,
  `result_callback`, metadata pass-through, `_resolve_callable`.
* Server: all endpoints, validation errors, end-to-end
  dispatch-then-poll via `TestClient`.
* Storage: capacity eviction, live-vs-terminal partition, per-session
  filtering.
* CompletionWriter: jsonl round-trip + batched-flush.

The full repo suite remains at zero pyright errors and zero ruff
errors with the new code in place.

## Near-term (independent of any one PR)

1. **Heartbeat + register protocol with a coordinator.**
   Today's `GatewayNode._heartbeat_loop` posts a node-state JSON to
   `<coordinator_url>/agentix/gateway/heartbeat`; the coordinator
   shape is not yet specified. Wire to a real rollout server (or
   keep the protocol coordinator-agnostic) once we have a concrete
   consumer.

2. **Streaming LLM proxy through the gateway.**
   `GatewayProxyHandler` mirrors the bridge's behaviour and asks the
   upstream for `stream=False`. Once the bridge gains true streaming
   (its ROADMAP item #1), the gateway adopts the same chunked path.

3. **Trace integration parity with the bridge.**
   Each session opens a `trace.span("gateway.session", session_id=...)`
   that spans the four lifecycle stages; LLM calls captured inside
   the session ride the abridge `llm.request` spans (already shipped
   in PR #37) so the trace tree shows session -> stage -> llm.

4. **Result back-pressure.** When the in-memory `SessionStore` is at
   capacity, terminal sessions are evicted in insertion order. Add
   an opt-in disk-backed eviction store so the gateway never loses a
   terminal result when callers haven't drained results yet.

5. **Process-pool stage isolation.** Polar's gateway uses separate
   worker pools so a slow INIT can't starve a hot RUNNING. Today
   we rely on Agentix's per-sandbox isolation; revisit if CPU-bound
   POSTRUN work (trajectory scoring, large records writes) starts
   competing with active generations.

## Medium-term

1. **Trainer-facing pause/resume callbacks.**
   Today's `pause()` / `resume()` is a hard gate. Add a configurable
   callback chain so a coordinator can opt into "drain in flight",
   "abort in flight", or "queue at READY" semantics.

2. **Replay sessions.** Pair with the bridge's `ReplayClient` (bridge
   ROADMAP item #3) to drive a session entirely from a previously
   captured `RecordStore` snapshot — useful for offline eval, RL
   buffer regression checks, and deterministic test runs.

3. **Session API for grouped rollouts.** Mirror
   `polar.gateway.session.SessionSpec` group ids: one parent
   `rollout_id`, many child sessions, so coordinators can ask "give
   me the records for rollout R" in one shot. Maps onto a new
   `SessionSpec.rollout_id` plus a `RecordStore.for_rollout(...)`
   accessor.

4. **Pluggable per-call upstream routing.** The host today binds one
   `OpenAICompatibleClient` per gateway. Add a `route(spec) ->
   ClientSpec` hook so different agents in different sessions can
   hit different upstreams from the same gateway node — useful when
   one cluster fronts vLLM + an external OpenAI key.

5. **GRPC surface.** Some coordinators prefer gRPC over HTTP+JSON.
   The dispatcher / session / record types are protocol-agnostic;
   the `server.py` HTTP layer is replaceable.

## Long-term

* **K8s operator.** Bundle this gateway as a deployment-axis
  `Deployment` so a `GatewayNode` itself can be created via the
  same `agentix-deployment-...` registry. Useful for autoscaling a
  fleet of gateway pods behind a coordinator.

* **OTel export of `gateway.session` spans by default.** When
  `agentix-trace-otel` is installed and an `OTEL_EXPORTER_OTLP_*`
  env var is set, register an `OTelTraceProcessor` automatically so
  cluster operators get gateway visibility without writing any
  code.

* **Replay buffer schema.** Settle on a stable JSONL / parquet
  schema for `CompletionRecord` + `SessionResult` so an offline
  trainer or evaluator can read gateway output without hand-rolling
  schema interrogation.

* **Multi-tenant gateways.** Today every session shares one
  upstream client. Multi-tenant gateways need namespace isolation
  (auth, rate limits, quota) on the HTTP surface.

## Non-goals

* **The gateway does not own the agent harness.** Agents live as
  importable callables (or as agent-plugin namespaces like
  `agentix.agents.mini_swe_agent`); the gateway just dispatches them.

* **The gateway does not generate trajectories.** Trajectory
  conversion is each agent plugin's responsibility (e.g. PR #39's
  `Trajectory` for mini-swe-agent). The gateway stores whatever the
  callable returns, plus the captured records.

* **The gateway does not authenticate upstream API calls inside
  the sandbox.** Credentials live on the host (in the
  `OpenAICompatibleClient` or its successors); the sandbox is
  always credential-free.
