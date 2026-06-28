# sidecars/

Host-side gateway sidecars that abridge forwards to. These are **standalone
vendored projects** — NOT uv workspace members, NOT part of the `agentix`
package. abridge core stays shape/protocol-blind; all protocol and ML logic
lives here, behind a localhost HTTP process.

- `cc_convert/` — Anthropic ↔ OpenAI translation sidecar (Rust core + axum
  binary + PyO3 wheel). abridge's `agentix.bridge.sidecars.cc_convert_sidecar(...)`
  preset launches the `cc_convert_sidecar` binary.
- `tito/` — TITO pretokenize + session-recording gateway (FastAPI, wraps
  Miles). Sits in front of an sglang / OpenAI-compatible backend and emits
  pretokenized RL rollout trajectories.

Each keeps its own build system and dependencies; nothing here is installed
into the core venv. Upstream attributions are preserved in each subtree
(`cc_convert/LICENSE-*`, `tito/VENDORED_MILES_AUDIT.md`).

## Status / planned refactor

Vendored as-is to get the sources in-tree; refactor follows.

- **cc_convert** ships as a Rust binary today. The plan is to drop the
  binary requirement and drive translation from code in-process (it already
  exposes a PyO3 Python package under `cc_convert/python/`), so abridge can
  call it without launching a separate process.
- **tito** runs as a FastAPI sidecar; its session/trajectory records are
  bridged onto the existing `/trace` channel (work in progress).
