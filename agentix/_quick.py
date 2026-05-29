"""One-call convenience for running a function in a fresh sandbox.

`quick_remote(fn, bundle=...)` collapses the common path — load a
deployment backend, open a `session`, connect a `RuntimeClient`, call
`remote(fn, ...)`, return the result — into a single await:

    result = await agentix.quick_remote(run, bundle=BUNDLE)

The full flow (`load_deployment` / `SandboxConfig` / `session` /
`RuntimeClient`) stays available when you need resource limits, env
vars, plugin namespaces, or to reuse one sandbox across many calls.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from agentix.deployment.base import SandboxConfig, SandboxResource, load_deployment, session
from agentix.runtime.client import RuntimeClient

R = TypeVar("R")


async def quick_remote(
    fn: Callable[..., R] | Callable[..., Awaitable[R]],
    *args: Any,
    bundle: str,
    image: str = "python:3.13-slim",
    deployment: str = "docker",
    env: dict[str, str] | None = None,
    platform: str | None = None,
    resource: SandboxResource | None = None,
    timeout: float = 300,
    **kwargs: Any,
) -> R:
    """Run ``fn(*args, **kwargs)`` in a throwaway sandbox and return its result.

    Loads the ``deployment`` backend, creates a sandbox from ``image`` +
    ``bundle``, runs ``fn`` once via ``RuntimeClient.remote``, then tears the
    sandbox down. ``bundle`` is the reference produced by ``agentix deploy``
    (for docker/podman the cache path; ``agentix deploy ... --format json``
    prints it as JSON).
    """
    deployment_cls = cast(Any, load_deployment(deployment))
    backend = deployment_cls()
    config = SandboxConfig(
        image=image,
        bundle=bundle,
        env=env,
        platform=platform,
        resource=resource,
    )
    async with session(backend, config) as sandbox:
        async with RuntimeClient(sandbox.runtime_url, timeout=timeout) as client:
            return await client.remote(fn, *args, **kwargs)


__all__ = ["quick_remote"]
