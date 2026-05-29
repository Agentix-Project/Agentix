"""SandboxProvider Protocol, the live Sandbox handle, and the plugin registry.

A provider is anything that creates / deletes / inspects a sandbox. Backends
ship in their own packages (`agentix-deployment-docker`, ...) and subclass
`SandboxProvider`: they implement the three lifecycle methods and inherit the
`session(...)` helper.

```toml
# downstream pyproject.toml
[project.entry-points."agentix.provider"]
fly = "agentix.provider.fly:FlyProvider"
```

```python
# downstream module
from agentix.provider.base import SandboxProvider

class FlyProvider(SandboxProvider):
    async def create(self, config): ...
    async def delete(self, sandbox_id): ...
    async def get(self, sandbox_id): ...
```

Typed user code imports the concrete provider directly
(`from agentix.provider.docker import DockerProvider`). The entry-point
registry exists only for the string-keyed boundaries — `agentix deploy <name>`
and `agentix plugin list` — not as a typed construction API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NewType, ParamSpec, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from agentix.provider._plugin import Registry

if TYPE_CHECKING:
    from agentix.runtime.client import RuntimeClient
    from agentix.runtime.shared.models import HealthResponse

P = ParamSpec("P")
R = TypeVar("R")

SandboxId = NewType("SandboxId", str)
"""Provider-side handle for a running sandbox container. Returned by
`SandboxProvider.create(...)` and threaded back through `delete(...)` /
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
    """Configuration a provider uses to provision a sandbox.

    Two artifacts, one sandbox. `bundle` is the generic Agentix
    bundle produced by `agentix build` — it carries the runtime server,
    user callables, and their Python deps under `/nix/runtime/`.
    `image` is the task-specific base (e.g. a SWE-bench task image, a
    customer environment image) the workload actually runs against.

    The provider makes the bundle's `/nix` tree appear at `/nix` in
    the task sandbox, then execs `/nix/runtime/bootstrap.sh` inside
    `image`'s filesystem (see `BUNDLE_RUNTIME_ENTRYPOINT`).
    """

    image: str = Field(
        description="Task base image — the container OS and task environment the "
        "workload runs in (e.g. `python:3.13-slim` or "
        "`swebench/task-django__django-12345:latest`). The provider mounts the "
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
    """Live sandbox handle — call `await sandbox.remote(fn, ...)` directly.

    The handle owns a `RuntimeClient` that connects lazily on the first
    `remote()` / `health()` call, so callers never wire one up by hand.
    `await sandbox.aclose()` (or `async with sandbox: ...`) closes that
    connection. The container itself is removed by the provider —
    `await provider.delete(sandbox.sandbox_id)`, or automatically by
    `async with provider.session(config) as sandbox: ...`.

    Register host-side plugin namespaces with `register_namespace(...)`
    before the first `remote()` (the connection plan is fixed at connect
    time).
    """

    sandbox_id: SandboxId
    runtime_url: str
    status: str
    _client: RuntimeClient | None = field(default=None, init=False, repr=False, compare=False)

    def _runtime_client(self) -> RuntimeClient:
        if self._client is None:
            from agentix.runtime.client import RuntimeClient as _RuntimeClient

            self._client = _RuntimeClient(self.runtime_url)
        return self._client

    def register_namespace(self, namespace: Any) -> None:
        """Register a host-side plugin namespace before the first remote call."""
        self._runtime_client().register_namespace(namespace)

    async def remote(
        self,
        fn: Callable[P, R] | Callable[P, Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Execute `fn(*args, **kwargs)` in this sandbox and return its result."""
        return await self._runtime_client().remote(fn, *args, **kwargs)

    async def health(self) -> HealthResponse:
        return await self._runtime_client().health()

    async def aclose(self) -> None:
        """Close the runtime connection (idempotent). Does not delete the container."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> Sandbox:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


@runtime_checkable
class SandboxProvider(Protocol):
    """Sandbox lifecycle management.

    Backends subclass this, implement the three lifecycle methods, and
    inherit the `session(...)` helper. They are typically registered as
    entry points in the `agentix.provider` group and constructed directly
    (often with a backend config object), e.g.
    `DockerProvider(DockerProviderConfig(...))`.
    """

    async def create(self, config: SandboxConfig) -> Sandbox: ...
    async def delete(self, sandbox_id: SandboxId) -> None: ...
    async def get(self, sandbox_id: SandboxId) -> SandboxInfo: ...

    @asynccontextmanager
    async def session(self, config: SandboxConfig) -> AsyncIterator[Sandbox]:
        """Scoped sandbox: created on entry; its client closed and the
        container deleted on exit.

            async with provider.session(SandboxConfig(...)) as sandbox:
                result = await sandbox.remote(fn, ...)
        """
        sandbox = await self.create(config)
        try:
            yield sandbox
        finally:
            await sandbox.aclose()
            await self.delete(sandbox.sandbox_id)


@runtime_checkable
class BundleMaterializer(Protocol):
    """Optional provider hook for `agentix deploy`.

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


# The plugin registry — one `agentix.provider` group. Backend dists add
# their own entry points. Tests can also `register_provider("fake", ...)`
# imperatively via the public helper below. This powers the string-keyed
# boundaries (`agentix deploy <name>`, `agentix plugin list`); typed code
# imports the concrete provider class directly.
_providers: Registry[type[SandboxProvider]] = Registry("agentix.provider")


def register_provider(name: str, cls: type[SandboxProvider]) -> None:
    """In-process provider registration. Test / dynamic use only —
    production backends are declared in their dist's `pyproject.toml`
    `[project.entry-points."agentix.provider"]`."""
    _providers.register(name, lambda: cls)


def providers() -> Registry[type[SandboxProvider]]:
    """The provider registry — for discovery (`.all()`) and the CLI.

    Typed user code should import the concrete provider class directly
    (`from agentix.provider.docker import DockerProvider`) rather than
    resolving a class from a string here.
    """
    return _providers
