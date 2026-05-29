"""Deployment Protocol + plugin registry.

A deployment backend is anything that creates / deletes / inspects a
sandbox. The framework treats them as plugins: each backend is a class
registered under the `agentix.deployment` entry-point group. Backends
ship in their own packages (`agentix-deployment-docker`,
`agentix-deployment-fly`, ...).

```toml
# downstream pyproject.toml
[project.entry-points."agentix.deployment"]
fly = "agentix_deployment_fly:FlyDeployment"
```

```python
# downstream module
from agentix.deployment import Deployment   # Protocol

class FlyDeployment:                          # no inheritance, structural type
    async def create(self, cfg): ...
    async def delete(self, sid): ...
    async def get(self, sid): ...
```

`load_deployment("fly")` works after the install with zero framework
changes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NewType, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from agentix.deployment._plugin import Registry

SandboxId = NewType("SandboxId", str)
"""Deployment-side handle for a running sandbox container. Returned by
`Deployment.create(...)` and threaded back through `delete(...)` /
`get(...)`."""


class SandboxResource(BaseModel):
    """Resource request for one sandbox."""

    cpu: float | None = Field(
        default=None,
        gt=0,
        description="Optional CPU count requested for the sandbox, e.g. 4 or 0.5.",
    )
    memory: int | str | None = Field(
        default=None,
        description=(
            "Optional memory limit requested for the sandbox. "
            "Strings use the container CLI unit syntax, e.g. `16g`."
        ),
    )
    gpu: int | None = Field(
        default=None,
        gt=0,
        description="Optional GPU count requested for the sandbox.",
    )

    @field_validator("memory")
    @classmethod
    def _validate_memory(cls, value: int | str | None) -> int | str | None:
        if value is None:
            return None
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("memory must be positive")
            return value
        if not value.strip():
            raise ValueError("memory must not be empty")
        return value


class SandboxConfig(BaseModel):
    """Configuration a deployment uses to provision a sandbox.

    Two artifacts, one sandbox. `bundle` is the generic Agentix
    bundle produced by `agentix build` — it carries the runtime server,
    user callables, and their Python deps under `/nix/runtime/`.
    `image` is the task-specific base (e.g. a SWE-bench task image, a
    customer environment image) the workload actually runs against.

    The deployment makes the bundle's `/nix` tree appear at `/nix` in
    the task sandbox, then execs `/nix/runtime/bootstrap.sh` inside
    `image`'s filesystem (see `BUNDLE_RUNTIME_ENTRYPOINT`).
    """

    image: str = Field(
        description="Task base image — the container OS and task environment the "
        "workload runs in (e.g. `python:3.13-slim` or "
        "`swebench/task-django__django-12345:latest`). The deployment mounts the "
        "bundle's `/nix` runtime tree into this image, so swapping the task image "
        "needs no bundle rebuild.",
    )
    bundle: str = Field(
        description=(
            "Agentix runtime bundle reference — the runtime server, your code, and "
            "the Python deps that overlay onto `image` (read-only at `/nix`). "
            "`agentix build` produces the portable tar; `agentix deploy <backend>` "
            "materializes it into this backend-native ref (for docker/podman, the "
            "cache path it prints). `image` is the task environment, `bundle` is "
            "the runtime that runs there — both are required."
        ),
    )
    platform: str | None = Field(
        default=None,
        description="Optional runtime platform for both the task image "
        "and bundle artifact, e.g. `linux/amd64` or `linux/arm64`.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional env vars passed to the sandbox container.",
    )
    resource: SandboxResource | None = Field(
        default=None,
        description="Optional resource request for CPU, memory, and GPU.",
    )


class SandboxInfo(BaseModel):
    sandbox_id: SandboxId
    runtime_url: str
    status: str = "running"


@dataclass
class MaterializedBundle:
    """Backend-specific bundle reference produced by `agentix deploy`."""

    bundle: str
    platform: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class Sandbox:
    """Live sandbox handle — `runtime_url` is what `RuntimeClient` connects to."""

    sandbox_id: SandboxId
    runtime_url: str
    status: str


@runtime_checkable
class Deployment(Protocol):
    """Sandbox lifecycle management. Structural type — backends don't
    inherit, they just implement the three methods.

    Backends are typically classes registered as entry points. Backend
    constructors may accept their own explicit config objects for direct
    use; `load_deployment` still returns the class so callers can choose
    how to instantiate it.
    """

    async def create(self, config: SandboxConfig) -> Sandbox: ...
    async def delete(self, sandbox_id: SandboxId) -> None: ...
    async def get(self, sandbox_id: SandboxId) -> SandboxInfo: ...


@runtime_checkable
class BundleMaterializer(Protocol):
    """Optional deployment backend hook for `agentix deploy`.

    `agentix build` produces the backend-neutral tar bundle. A materializer
    turns that portable artifact into the backend-native reference that
    `SandboxConfig.bundle` should carry for later sandbox creation. That
    reference is backend-side state; the sandbox still sees the runtime
    at the fixed in-container path `/nix`.
    """

    async def materialize_bundle(
        self,
        bundle: Path,
        *,
        name: str | None = None,
        platform: str | None = None,
    ) -> MaterializedBundle: ...


# The plugin registry — one `agentix.deployment` group. Backend dists add
# their own entry points. Tests can also `register_deployment("fake", ...)`
# imperatively via the public helper below.
_deployments: Registry[type[Deployment]] = Registry("agentix.deployment")


def register_deployment(name: str, cls: type[Deployment]) -> None:
    """In-process deployment registration. Test / dynamic use only —
    production deployments are declared in their dist's `pyproject.toml`
    `[project.entry-points."agentix.deployment"]`."""
    _deployments.register(name, lambda: cls)


def load_deployment(name: str) -> type[Deployment]:
    """Return the deployment class registered under `name`.

    Raises `KeyError` (with available names) if no backend claims that
    name, or re-raises the loader's exception if the backend's import
    fails.
    """
    return _deployments.get(name)


def deployments() -> Registry[type[Deployment]]:
    """The underlying registry — for tests and introspection."""
    return _deployments


@asynccontextmanager
async def session(
    deployment: Deployment,
    config: SandboxConfig,
) -> AsyncIterator[Sandbox]:
    """Scoped sandbox: created on entry, deleted on exit.

    Free function instead of a Deployment method so the Protocol stays
    minimal (three methods); structural backends don't have to inherit
    a helper class.
    """
    sandbox = await deployment.create(config)
    try:
        yield sandbox
    finally:
        await deployment.delete(sandbox.sandbox_id)
