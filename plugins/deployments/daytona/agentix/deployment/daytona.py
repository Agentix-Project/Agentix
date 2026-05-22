"""Daytona deployment backend — stub.

Daytona (https://www.daytona.io/) runs managed sandboxes from OCI
images. The integration shape is captured below; the actual REST
client is the next item on the deploy roadmap. The class exists so
`agentix deploy daytona` fails with a clear error and so callers
can write code against the real Protocol contract in advance.

API key comes from `DAYTONA_API_KEY` env. No constructor arguments —
plugin loaders instantiate this with `cls()` uniformly.
"""

from __future__ import annotations

import os

from agentix.deployment.base import Sandbox, SandboxConfig, SandboxId, SandboxInfo


class DaytonaDeployment:
    """Sandbox CRUD via Daytona (pending integration)."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("DAYTONA_API_KEY")

    async def create(self, config: SandboxConfig) -> Sandbox:  # noqa: ARG002
        raise NotImplementedError(
            "DaytonaDeployment is not wired yet. The CLI surface exists so "
            "you can plan against it; the Daytona REST integration is the "
            "next item on the deploy roadmap."
        )

    async def delete(self, sandbox_id: SandboxId) -> None:  # noqa: ARG002
        raise NotImplementedError("DaytonaDeployment.delete: see create()")

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:  # noqa: ARG002
        raise NotImplementedError("DaytonaDeployment.get: see create()")
