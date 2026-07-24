# agentix-provider-cape

CAPE provider backend for
[Agentix](https://github.com/Agentix-Project/Agentix).

CAPE is a lease-based GPU capacity pool: a submitter-side `cape` CLI
sends workload requests to a controller, which schedules them onto pool
nodes with strong per-request isolation. Requests sharing a
`--session-key` land in the same warm session (RUNNING_COMMAND ↔
CACHED_IDLE — a *serial* model, one command at a time per session).
This backend maps one Agentix sandbox onto exactly one long-lived CAPE
request inside a per-sandbox session: the request bind-mounts the
bundle's `/nix` tree read-only and a per-sandbox meta directory
read-write at `/agentix-meta`, creates its workspace, writes an
`AGENTIX_ENDPOINT <host> <port>` marker file into the meta dir, and
execs `/nix/runtime/bootstrap.sh`. The provider discovers the endpoint
by polling the *host side* of that marker file, interleaved with
`cape status` (it never submits a second command into the session, so
it does not depend on same-session concurrency or on cross-request
workspace persistence), then health-checks `GET /health` on the
discovered endpoint.

## Contract status — verified against the real CLI

**The CLI surface implemented here (`cape run/status/logs/cancel`) has
been verified item by item against the CAPE source.** The original
checklist items now all have source-verified answers; `_CapeCli` in
`agentix/provider/cape.py` remains the single class that knows the CLI.

What matched the previously assumed contract:

* Verb set: `run` / `status` / `logs` / `cancel` all exist with
  `--controller-url` and `--token` on each. (`cape wait` also exists
  but is deliberately unused: its process exit code is 0 both on
  timeout and on null-exit-code terminal states, so only the status
  JSON `state` field can be trusted.)
* `cape run` flags: `--image`, `--gpus`, `--cpu-cores`, `--memory-gb`,
  `--isolation-policy`, `--runtime-adapter`,
  `--max-duration-seconds`, `--session-key`,
  `--bind src:dst[:ro|rw]`, `--env K=V`, and the `-- <workload>`
  remainder are all real, with matching semantics.
* Without `--wait`, `cape run` prints exactly one request id on
  stdout. Real ids are `req-%06d`; the provider matches
  `req-\d{6,}`.
* `cape status` prints one JSON object on stdout (the human-readable
  state line goes to stderr); the terminal-state set is exactly
  {COMPLETED, FAILED, CANCELLED, EXPIRED, LOST, INFEASIBLE}; the
  `request_id` echo the provider cross-checks is always present. The
  status JSON has **no stdout/stderr fields** — output only travels
  through `cape logs`.
* `cape cancel` takes `--reason`, is an idempotent no-op on already
  terminal requests, and rc=0 confirms the cancel. HTTP errors are a
  single stderr JSON line (`{"status": <code>, "error": ...}`) with
  rc=1; argv errors exit 2.
* The token can **only** be passed as `--token` on argv — the real CLI
  reads no env var and no token file, and there is no mint endpoint.
  This provider's token-file/env indirection is client-side
  convenience feeding that flag; the value is redacted from all error
  text, but `ps` visibility cannot be avoided from the CLI side
  (direct HTTP with the `X-CAPE-Token` header is the only
  alternative).
* The session model is serial (RUNNING_COMMAND ↔ CACHED_IDLE), as
  assumed; one sandbox = one request = one unique session key remains
  the right mapping.

What the verification changed:

* **`--task-id` is required** by the real parser (argparse exits 2
  without it). The provider now passes the sandbox id.
* **`--pool` / `--user` are required.** `CapeProviderConfig.pool` and
  `.user` are checked at first use and fail fast with a clear error
  instead of an argparse usage dump.
* **`--workspace` does not exist** — the real parser rejects it with
  `unrecognized arguments`, rc=2. The flag is gone; the boot script
  now creates the workspace itself (`mkdir -p <ws> && cd <ws>`). For
  the same reason `--cwd` is no longer passed: no component creates
  the directory, and every CAPE runtime adapter fails at process
  start on a missing cwd.
* **Endpoint discovery was redesigned.** `cape logs` succeeds for
  RUNNING requests but returns *empty* output until the command exits:
  the node agent reports stdout/stderr once, after reaping the
  process — CAPE has no streaming/incremental log channel and no
  native endpoint query (the status `ports[]` field is a request-spec
  echo that never reaches the node). The former stdout-marker +
  logs-polling design could therefore never see the marker of a
  never-exiting runtime server. Discovery now uses a **marker file
  through a rw bind**: each sandbox gets `<meta_root>/<sandbox_id>`
  (created before submit, removed on delete), bound at
  `/agentix-meta`; the boot script writes the
  `AGENTIX_ENDPOINT <host> <port>` line to `/agentix-meta/endpoint`;
  the provider polls the host side of that file, interleaved with
  `cape status` terminal-state checks. Logs are fetched only on
  terminal states — where they *are* populated — to enrich error
  messages.
* **Cancel treats a controller 404 as already-gone.** After a
  journal-less controller restart the request id is unknown; the old
  behavior kept bookkeeping (and its port) forever. A 404 in the
  CLI's stderr JSON now confirms deletion.

Later upstream CLI changes tracked:

* **`--gpu-mode` was removed from `cape run`.** Requests now inherit
  exclusive/shared GPU behavior from the target *pool*; there is no
  per-request override. Sending the old flag is an unknown-argument
  error (argparse exits 2 with a usage message before anything is
  submitted), so the provider no longer emits it.
* **`cape status` gained cotenancy fields**: `gpu_mode` (derived,
  `null` | `"exclusive"` | `"shared"`), `gpu_cotenant_count`, and
  `gpu_cotenant_counts`. The provider's status parsing only reads
  `state` and the `request_id` echo, so the new fields are tolerated
  (ignored) without changes.

### Networking: pin a host-network runtime adapter

The client-assigned-port scheme (`AGENTIX_BIND_PORT` plus a direct
probe of `<host>:<port>`) requires the workload to share the host's
network stack. CAPE's apptainer and bubblewrap adapters do not unshare
the network namespace, so they work; **CAPE's podman adapter runs
without host networking**, so an endpoint bound inside it is
unreachable and this provider cannot work on podman-only pools. The
default `runtime_adapter` stays `None` (pool default), but deployments
must pin `apptainer` or `bubblewrap` (or ensure the pool default is a
host-network adapter). `local-process` shares everything and suits
CPU-only local validation; with it, bind specs are no-ops, and the
boot script falls back to the host-side meta dir and bundle paths
(valid because that adapter shares the host filesystem).

## Install

```bash
pip install agentix-provider-cape
```

Set `CAPE_CONTROLLER_URL` and a token (`CAPE_TOKEN_FILE` pointing at a
`chmod 600` file, or the `CAPE_TOKEN` env var). `CAPE_BINARY` overrides
which `cape` binary is invoked. The token is re-read on every operation,
so rotation needs no restart, and its literal value is redacted from
any error output.

## Use

```python
from agentix import SandboxConfig
from agentix.provider.cape import CapeProvider, CapeProviderConfig

provider = CapeProvider(
    CapeProviderConfig(
        controller_url="https://cape-controller.example:8443",
        pool="coding-agent-gpu",          # required (`cape run --pool`)
        user="alice",                     # required (`cape run --user`)
        meta_root="/mnt/shared/agentix-meta",  # required (endpoint discovery)
        runtime_adapter="apptainer",      # pin a host-network adapter
        token_file="~/.config/cape/token",
    )
)
config = SandboxConfig(
    image="docker://nvcr.io/org/sandbox-runtime@sha256:...",
    bundle="/mnt/shared/agentix-bundles/sha256-.../",  # node-visible path
    resource={"gpu": 1},
)

async with provider.session(config, call_deadline=1800) as sandbox:
    result = await sandbox.remote(run, input="hello")
```

Backend-specific notes:

* **`SandboxConfig.bundle` is an opaque node-visible path** to an
  already-extracted bundle tree (its `nix/`... contents are bind-mounted
  read-only at `/nix`). How the bundle tree gets onto CAPE nodes —
  shared filesystem, prior upload, a CAPE-side template — is an **open
  question**; this iteration deliberately ships no `BundleDeployer` /
  `agentix deploy cape` until bundle transport is decided.
* **`meta_root` must be visible to both the submitter and the pool
  nodes** — the same shared-filesystem assumption `bundle` already
  makes, so it adds no new deployment requirement. Each sandbox gets
  its own `<meta_root>/<sandbox_id>` directory (rw-bound at
  `/agentix-meta`), which `delete()` removes after a confirmed cancel.
* **`url_template`** controls how the runtime URL is built from the
  discovered `host`/`port`. The default `http://{host}:{port}` assumes
  the submitter can route to pool nodes directly; behind an SSH tunnel,
  set `url_template="http://127.0.0.1:{port}"` and forward the port
  yourself.
* Health probing uses a raw TCP socket with a minimal HTTP request —
  never a proxy-aware HTTP client — so corp-proxy env vars cannot
  poison loopback/tunnel probes.
* `providers().get("cape")` resolves after `uv sync` / `pip install`.

## Operational notes and known limitations

* **Residual submit window.** `create()` shields the initial `cape run`
  so an external cancellation still harvests the request id and cancels
  the request. But if the submitter process is killed (or the CLI dies)
  in the instant after the controller accepted the request and before
  its id was printed, that request cannot be cancelled from this side —
  it runs until `--max-duration-seconds`. Session keys are always
  prefixed `agentix-cape-...`, so server-side/admin tooling can find and
  reclaim orphaned Agentix sessions by that naming convention.
* **Bookkeeping is in-process only.** The `sandbox_id → request_id` map
  lives in the provider instance; if the submitter process crashes
  after `create()`, a fresh provider cannot see or delete the old
  sandbox (`delete()` of an unknown id is a silent no-op). Recovery is
  server-side cleanup by the `agentix-cape-...` session-key convention,
  or waiting out `--max-duration-seconds`. The real CLI has no
  list/query-by-session verb to rebuild the map from.
* **Delete retries instead of leaking.** `delete()` drops bookkeeping
  only after the controller confirms the cancel (`cape cancel` rc=0) or
  reports the request as unknown (HTTP 404 — e.g. after a journal-less
  controller restart, when the request is gone anyway); any other
  failed cancel logs a warning and keeps the record so calling
  `delete()` again retries it.
* **Runtime ports.** Each live sandbox of one provider instance gets a
  distinct `AGENTIX_BIND_PORT` from
  `[runtime_port_base, runtime_port_base + runtime_port_span)`, so
  sandboxes of the same provider can never answer each other's health
  probes (same node or same SSH tunnel). Port collisions with *other*
  submitters or users on the same node cannot be reserved from this
  side — the real contract has no controller-side port brokering.

## License

MIT — see the repository root
[LICENSE](https://github.com/Agentix-Project/Agentix/blob/master/LICENSE).
