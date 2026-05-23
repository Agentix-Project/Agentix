# hello-world

Minimal Agentix example with one Python file and one Nix dependency.

`main.py` defines `hello()`, which runs `rg --version`. The same function
is called once on the host and once inside an Agentix sandbox. The sandbox
gets `rg` from `default.nix`, which `agentix build` merges into
`/nix/runtime/bin`.

## Run

```bash
uv sync
uv run agentix build . --format oci-image
uv run main.py
```

## Flow

1. `uv sync` installs this example plus the local Agentix packages from
   `[tool.uv.sources]`.
2. `uv run agentix build . --format oci-image` builds the runtime
   bundle image from this project. It installs `main.py` into the
   bundle and applies `default.nix`, so `rg` is available in the
   sandbox runtime.
3. `uv run main.py` starts a local Docker sandbox using the built
   `hello-world` bundle, then prints the host result and sandbox result.

Docker must be running before the build and sandbox run.
