"""Shared models for Agentix.

Closures are static per sandbox: deployment mounts each closure under /mnt,
runtime scans /mnt, imports each closure's declared Python package, and
binds its dispatcher in-process. No subprocesses, no UDS, no /load, no
caller-chosen namespaces — the Python package path is the unique identity.

Wire: `POST /_remote` body RemoteRequest -> RemoteResponse.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Closure manifest (shipped inside the closure image) ───────────

AGENTIX_CLOSURE_ABI = 1
"""Protocol version of the closure convention. Runtime ignores closures whose
manifest declares a different value. Bump on hard breaks (path layout,
manifest schema, dispatch ABI)."""


class ClosureManifest(BaseModel):
    """Static metadata shipped at `/nix/entry/manifest.json` inside a closure
    image. Presence of this file is what marks a `/mnt/<ns>` mount as an
    Agentix closure — runtime ignores anything without one.

    `package` is the Python import path the runtime imports at startup to
    obtain the closure's Dispatcher (via `<package>._register.register()`).
    """

    abi: int
    name: str
    version: str
    package: str = Field(
        description="Python import path of the closure package, e.g. 'agentix_closures.claude_code'."
    )
    description: str | None = None
    kind: str | None = Field(
        default=None,
        description="Optional, purely informational (e.g. 'agent', 'dataset', 'tool'). Runtime ignores.",
    )

    model_config = {"extra": "allow"}


# ── Runtime server wire types ─────────────────────────────────────


class ClosureInfo(BaseModel):
    """One entry in GET /closures. `manifest.package` is the closure identity;
    `path` is the on-disk mount, exposed for debugging only.
    """

    path: str
    manifest: ClosureManifest


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


# ── Remote-call wire ─────────────────────────────────────────────


class RemoteRequest(BaseModel):
    """POST /_remote body. `package` is the closure's Python import path
    (e.g. 'agentix_closures.claude_code'); `method` is a stub name bound
    by that closure's Dispatcher.
    """

    package: str
    method: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class RemoteError(BaseModel):
    type: str
    message: str
    traceback: str | None = None


class RemoteResponse(BaseModel):
    """POST /_remote response."""

    ok: bool
    value: Any = None
    error: RemoteError | None = None


# ── Runtime I/O primitives (exec / upload / download) ────────────


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


# ── Deployment ────────────────────────────────────────────────────


class SandboxConfig(BaseModel):
    image: str = Field(description="Base Docker/OCI image the sandbox runs on (the task environment)")
    runtime: str = Field(description="Runtime closure image ref")
    closures: list[str] = Field(
        default_factory=list,
        description="Closures to mount, by image ref. Each closure's identity "
        "comes from its manifest's `package` field — there are no caller-chosen "
        "namespaces. Two images shipping the same package collide and the second "
        "is skipped with a warning.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional env vars passed to the sandbox container (and therefore "
            "visible to the runtime + all closures)."
        ),
    )


class SandboxInfo(BaseModel):
    sandbox_id: str
    runtime_url: str
    status: str = "running"
