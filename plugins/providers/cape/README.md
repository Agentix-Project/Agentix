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
bundle's `/nix` tree read-only, prints an `AGENTIX_ENDPOINT <host>
<port>` marker line to stdout, and execs `/nix/runtime/bootstrap.sh`.
The provider discovers the endpoint by polling `cape status` and
`cape logs` for that marker (it never submits a second command into the
session, so it does not depend on same-session concurrency or on
cross-request workspace persistence), then health-checks `GET /health`
on the discovered endpoint.

## Contract status — ASSUMED CLI, not verified

**The CAPE CLI verb surface implemented here
(`cape run/status/logs/cancel`, their flags, the request-id stdout
shape, and the `cape status` JSON schema) is an *assumed* contract.**
It was reverse-documented from a sibling project's adapter of the
same CLI, whose authors record that the official CAPE CLI was never
obtained: the contract has only ever been exercised
against fakes and local emulators. Treat every invocation in
`agentix/provider/cape.py` (`_CapeCli` is the single class that knows
the CLI) as a hypothesis to confirm.

Verification checklist for when the real CLI arrives:

- [ ] Run `cape --help` (and per-verb `--help`) and diff the verb set
      against `run` / `status` / `logs` / `cancel`.
- [ ] Compare the `cape run` argv table (`--controller-url`, `--token`,
      `--pool`, `--user`, `--workspace`, `--image`, `--gpus`,
      `--cpu-cores`, `--memory-gb`, `--gpu-mode`, `--isolation-policy`,
      `--runtime-adapter`, `--max-duration-seconds`, `--cwd`,
      `--session-key`, `--bind`, `--env`, `-- <workload>`) flag by
      flag, including the "stdout is exactly one `req-...` line" parse
      rule.
- [ ] Compare the `cape status` JSON schema: the `state` field, the
      terminal-state set (COMPLETED / FAILED / CANCELLED / EXPIRED /
      LOST / INFEASIBLE), and the `request_id` echo this provider
      cross-checks.
- [ ] Confirm `cape logs` can return the stdout of a still-RUNNING
      request — endpoint discovery depends on it. This assumption is
      *new in this provider* (the reference adapter only called logs on
      terminal requests).
- [ ] Confirm the session model is serial (RUNNING_COMMAND ↔
      CACHED_IDLE). This provider deliberately submits only one request
      per session, so it works either way — but tooling built on top
      must not assume same-session concurrency.
- [ ] Confirm how the token is passed — `--token` on argv is visible in
      process listings; prefer an env var or token-file option if the
      real CLI supports one.
- [ ] Confirm how a workload's node/port can be discovered — the
      stdout-marker dance here exists only because the assumed contract
      has no native endpoint query.

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
        pool="coding-agent-gpu",
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
  or waiting out `--max-duration-seconds`. The assumed CLI has no
  list/query-by-session verb to rebuild the map from.
* **Delete retries instead of leaking.** `delete()` drops bookkeeping
  only after the controller confirms the cancel (`cape cancel` rc=0); a
  failed cancel logs a warning and keeps the record so calling
  `delete()` again retries it.
* **Runtime ports.** Each live sandbox of one provider instance gets a
  distinct `AGENTIX_BIND_PORT` from
  `[runtime_port_base, runtime_port_base + runtime_port_span)`, so
  sandboxes of the same provider can never answer each other's health
  probes (same node or same SSH tunnel). Port collisions with *other*
  submitters or users on the same node cannot be reserved from this
  side — that is a known limitation of the assumed contract (no
  controller-side port brokering).

## License

MIT — see the repository root
[LICENSE](https://github.com/Agentix-Project/Agentix/blob/master/LICENSE).
