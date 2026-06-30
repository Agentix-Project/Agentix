# sidecars/

Host-side gateway sidecars that abridge forwards to. These are **standalone
vendored projects** — NOT uv workspace members, NOT part of the `agentix`
package. abridge core stays shape/protocol-blind; all protocol logic lives
here, behind a localhost HTTP process.

- `cc_convert/` — Anthropic ↔ OpenAI translation sidecar (Rust core + axum
  binary + PyO3 wheel). abridge's `agentix.bridge.sidecars.cc_convert_sidecar(...)`
  preset launches the `cc_convert_sidecar` binary.

The TITO pretokenize + session-recording gateway used to live here; it is now a
first-class Agentix plugin at `plugins/tito` (`import agentix.tito`), natively
implemented with no vendored code.

Each sidecar keeps its own build system and dependencies; nothing here is
installed into the core venv. Upstream attributions are preserved in each
subtree (`cc_convert/LICENSE-*`).

## Status / planned refactor

Vendored as-is to get the sources in-tree; refactor follows.

- **cc_convert** ships as a Rust binary today. The plan is to drop the
  binary requirement and drive translation from code in-process (it already
  exposes a PyO3 Python package under `cc_convert/python/`), so abridge can
  call it without launching a separate process.
