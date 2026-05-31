# hello-world

Minimal Agentix example with one Python file and one Nix dependency.

`main.py` defines `hello()`, which runs `rg --version`. The same function
is called once on the host and once inside an Agentix sandbox. The sandbox
gets `rg` from `default.nix`, which `agentix build` merges into
`/nix/runtime/bin`.

## Run (local Docker)

```bash
uv sync
uv run agentix build . --output dist/hello-world.bundle.tar
BUNDLE=$(uv run agentix deploy docker dist/hello-world.bundle.tar | awk -F' -> ' '/^bundle -> /{print $2}')
uv run python main.py --bundle "$BUNDLE" # defaults to --provider docker
```

`main.py` accepts `--provider`, `--image`, and `--bundle`. By default
it loads the Docker backend and runs the cache path that `agentix deploy`
produced.

## Run on an HPC host (apptainer)

Build a portable tar bundle and run with the apptainer backend:

```bash
uv run agentix build . --output /path/to/hello-world.tar
uv run main.py \
    --provider apptainer \
    --bundle /path/to/hello-world.tar \
    --image docker://python:3.13-slim
```

The backend pulls the task image with `apptainer pull` on first use and
caches it under `$AGENTIX_APPTAINER_CACHE` (default
`~/.cache/agentix/apptainer/`).

## Flow

1. `uv sync` installs this example plus the local Agentix packages from
   `[tool.uv.sources]`.
2. `uv run agentix build .` builds the portable runtime bundle tar.
   The build installs `main.py` into the bundle and applies
   `default.nix`, so `rg` is available in the sandbox runtime.
3. `agentix deploy docker|podman` deploys the tar to a
   Docker-compatible backend (local-extract form).
4. `uv run main.py [--provider ...]` starts a sandbox with the
   selected backend, then prints the host result and sandbox result.
