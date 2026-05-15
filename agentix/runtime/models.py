"""Runtime transport wire types.

Lives under `agentix.runtime.models` because every type here is part of the
HTTP / Socket.IO surface between `RuntimeClient` (orchestrator side) and the
runtime server (sandbox side). Both client and server import from here;
sibling subpackages (`runtime/client/`, `runtime/server/`) depend on this
module but not on each other.

Cross-cutting concepts that aren't wire types — `ClosureManifest`,
`SandboxConfig`, `SandboxInfo`, `AGENTIX_CLOSURE_ABI` — stay in
`agentix.models` since they're consumed by closure authors / deployment
code that doesn't touch the runtime transport.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentix.models import ClosureManifest

# ── Runtime introspection (GET /closures, /health) ──────────────────


class ClosureInfo(BaseModel):
    """One entry in GET /closures. `manifest.package` is the closure identity;
    `path` is the on-disk mount, exposed for debugging only.
    """

    path: str
    manifest: ClosureManifest


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


# ── Remote-call wire (POST /_remote, plus Socket.IO stream/bidi events) ─


class RemoteRequest(BaseModel):
    """POST /_remote body. `package` is the closure's Python import path
    (e.g. 'agentix_closures.claude_code'); `method` is a stub name bound
    by that closure's Dispatcher.

    `call_id` is an optional rollout correlation key; the dispatcher pins
    it into a contextvar so trace events emitted from inside the impl
    inherit it automatically. Stream / bidi calls carry their own
    `call_id` on the Socket.IO wire — same semantic.
    """

    package: str
    method: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class RemoteError(BaseModel):
    type: str
    message: str
    traceback: str | None = None


class RemoteResponse(BaseModel):
    """POST /_remote response."""

    ok: bool
    value: Any = None
    error: RemoteError | None = None


# ── Log + trace Socket.IO events ────────────────────────────────────


class LogRecord(BaseModel):
    """One log line forwarded over the Socket.IO `log` event.

    Subscribers receive these whenever the runtime (or any closure logger
    under the `agentix.*` tree) emits a logging record.
    """

    level: str          # e.g. "INFO", "WARNING"
    name: str           # logger name
    message: str        # formatted message
    timestamp: float    # record.created — seconds since epoch


class TraceEvent(BaseModel):
    """One semantic event from a rollout, broadcast via the `trace` Socket.IO
    event. Closures emit these through `agentix.trace.emit(...)` to record
    LLM calls, tool invocations, rewards, or arbitrary checkpoint markers.

    The `call_id` correlates events to a specific rollout — for unary HTTP
    calls it can be set on `RemoteRequest.call_id`; for stream/bidi it's the
    `call_id` already on the Socket.IO wire frame. The dispatcher pins it
    into a contextvar before invoking the impl so `trace.emit()` picks it
    up automatically.
    """

    kind: str                       # e.g. "llm_request", "llm_response", "tool_call", "reward"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float                # emit-time seconds since epoch
    call_id: str | None = None      # rollout / call correlation key
    source: str | None = None       # closure package or "runtime" that emitted this


# ── Runtime I/O primitives (exec / upload / download) ───────────────


class ExecRequest(BaseModel):
    command: str
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: float | None = None
    max_output: int | None = Field(
        default=None,
        description="Cap on stdout/stderr bytes for buffered exec. Default: 10 MiB.",
    )
    paths_from: list[str] | None = Field(
        default=None,
        description=(
            "Python package paths of loaded closures whose `bin/` should be prepended "
            "to PATH for this command. Default: PATH is the task image's default, "
            "closure bins do not shadow it. Use ['agentix_closures.<name>'] or ['*'] "
            "when you explicitly want a closure's tools on PATH."
        ),
    )


class ExecResponse(BaseModel):
    exit_code: int
    stdout: str
    stderr: str


class UploadResponse(BaseModel):
    path: str
    size: int
