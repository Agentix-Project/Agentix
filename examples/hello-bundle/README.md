# hello-bundle

The smallest possible Agentix bundle — the fixture for the
`agentix build` end-to-end test.

```sh
agentix build examples/hello-bundle
```

It declares the framework (`agentixx`) plus one sandbox-side plugin
(`agentix-runtime-basic`), so building it exercises the whole pipeline:
the Nix toolchain, a `uv sync`'d venv, and a plugin-contributed system
closure merged into `/nix/runtime`.
