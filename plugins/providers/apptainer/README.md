# agentix-provider-apptainer

Apptainer (formerly Singularity) provider backend for Agentix. Targets
HPC and shared-cluster environments where the Docker daemon is
unavailable but `apptainer` is installed and unprivileged-friendly.

Registers `apptainer` in the `agentix.provider` entry-point group, so
`providers().get("apptainer")` resolves after `pip install
agentix-provider-apptainer`.

## Inputs

* `SandboxConfig.bundle` — path to a tar bundle produced by
  `agentix build`. The tar's `nix/` tree is extracted once
  per bundle digest into a process-local scratch directory and reused
  across sandboxes that share the bundle.
* `SandboxConfig.image` — task image. Apptainer-native references
  (`docker://repo/image`, `path/to/task.sif`, `library://...`,
  `oras://...`) all work; the backend passes the reference through
  unchanged. The image is converted to a SIF the first time it is seen
  (cached in `$AGENTIX_APPTAINER_CACHE` or `~/.cache/agentix/apptainer/`).
* `SandboxConfig.platform` — ignored (apptainer runs the host
  architecture; cross-arch is not supported in this backend).
* `SandboxConfig.env` — passed through as `APPTAINERENV_*` vars.

## How it runs

For each sandbox:

1. Extract the bundle tar to `<cache>/bundles/<digest>/nix/` (skipped if
   already populated).
2. Resolve `image` to a local `.sif` (cached by image ref hash).
3. Pick a free TCP port on `127.0.0.1` and pass it via
   `AGENTIX_BIND_PORT`.
4. Spawn `apptainer exec` with:
   * `--bind <cache>/bundles/<digest>/nix:/nix:ro`
   * `--userns --no-init --writable-tmpfs --cleanenv` (overridable
     via `AGENTIX_APPTAINER_FLAGS`)
   * `--env AGENTIX_BIND_PORT=<port>` (+ any `config.env`)
   * `<sif>` as the rootfs
   * a `/bin/sh -c` bootstrap that prepends `/nix/runtime/{venv/bin,bin,…}`
     execs `/nix/runtime/bootstrap.sh`, which preps
     `PATH`/`LD_LIBRARY_PATH`/etc. and hands off to uvicorn.
5. Poll `http://127.0.0.1:<port>/health` until it returns 200.
6. Return a `Sandbox` whose `runtime_url` points at that port.

`delete()` signals the spawned `apptainer exec` process, which in turn
takes down the runtime server and the container kernel.

Apptainer shares the host network namespace by default, so the runtime
server's port is directly reachable on `localhost` without `-p` style
publishing — there is no per-sandbox network setup.

## Environment

* `AGENTIX_APPTAINER_BIN` — override the `apptainer` binary path
  (defaults to whichever `apptainer` is on `PATH`).
* `AGENTIX_APPTAINER_CACHE` — scratch directory for extracted bundles
  and converted SIFs (defaults to `~/.cache/agentix/apptainer/`).
* `AGENTIX_APPTAINER_FLAGS` — whitespace-separated isolation flags
  passed to `apptainer exec`. Defaults to `--userns --no-init
  --writable-tmpfs`, which works on capability-restricted hosts. On a
  permissive host you can tighten isolation with e.g.
  `AGENTIX_APPTAINER_FLAGS="--containall --no-init --writable-tmpfs"`.

## Running examples

The cookbook examples (`examples/hello-world/`, `examples/run-claude-code/`,
`examples/run-mini-swe-agent/`) all accept a `--provider` flag — pass
`apptainer` to use this backend:

```sh
python examples/hello-world/main.py \
    --provider apptainer \
    --bundle /path/to/hello-world.tar \
    --image docker://python:3.13-slim
```

See `tests/test_apptainer_provider.py` for the local unit suite (uses
a recorded apptainer CLI; no real container runtime needed).
