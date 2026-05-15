"""Closure ABI + sandbox / deployment models.

These are the cross-cutting types: the closure-image contract (everyone
who builds or runs a closure depends on `ClosureManifest`) and the
top-level sandbox/deployment config that orchestrators hand to a
`Deployment`. Runtime transport / wire types live in
`agentix.runtime.models` instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

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

    model_config = {"extra": "allow"}


# ── Deployment ────────────────────────────────────────────────────


class SandboxConfig(BaseModel):
    image: str = Field(description="Base Docker/OCI image the sandbox runs on (the task environment)")
    runtime: str = Field(description="Runtime closure image ref")
    closures: list[str] = Field(
        default_factory=list,
        description=(
            "Closures to mount. Accepts docker image refs (strings) or any object "
            "exposing a string `__image__` attribute — typically the closure's "
            "imported Python package, e.g. `closures=[claude_code, mock_agent]`. "
            "Modules are resolved to their `__image__` at validation; the stored "
            "list is always strings. Each closure's runtime identity still comes "
            "from its manifest's `package` field — there are no caller-chosen "
            "namespaces."
        ),
    )
    env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional env vars passed to the sandbox container (and therefore "
            "visible to the runtime + all closures)."
        ),
    )

    @field_validator("closures", mode="before")
    @classmethod
    def _resolve_closure_specs(cls, v: Any) -> Any:
        """Accept ``list[str | <obj with __image__>]`` and normalise to list[str]."""
        if not isinstance(v, list):
            return v  # pydantic will reject below
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
                continue
            img = getattr(item, "__image__", None)
            if isinstance(img, str) and img:
                out.append(img)
                continue
            raise ValueError(
                f"closure spec {item!r} must be a docker-image-ref string or "
                f"an object with a non-empty string `__image__` attribute "
                f"(e.g. a closure's Python package module)"
            )
        return out


class SandboxInfo(BaseModel):
    sandbox_id: str
    runtime_url: str
    status: str = "running"
