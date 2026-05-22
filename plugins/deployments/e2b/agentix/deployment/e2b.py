"""E2B deployment backend — stub.

E2B (https://e2b.dev/) hosts ephemeral sandboxes seeded by their own
"template" image format. A bundle image needs to be published as an
E2B template (`e2b template build`) before it can be deployed here.
The class exists so `agentix deploy e2b` fails with a clear error
and so callers can write code against the real Protocol contract.

Config comes from env: `E2B_API_KEY` and `E2B_TEMPLATE_ID`. No
constructor arguments — plugin loaders instantiate this with `cls()`.
"""

from __future__ import annotations

import os

from agentix.deployment.base import Sandbox, SandboxConfig, SandboxId, SandboxInfo


class E2BDeployment:
    """Sandbox CRUD via E2B (pending integration)."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("E2B_API_KEY")
        self._template_id = os.environ.get("E2B_TEMPLATE_ID")

    async def create(self, config: SandboxConfig) -> Sandbox:  # noqa: ARG002
        raise NotImplementedError(
            "E2BDeployment is not wired yet. E2B's template system means a "
            "bundle image has to be published as a template first; the "
            "build pipeline + API integration are on the deploy roadmap."
        )

    async def delete(self, sandbox_id: SandboxId) -> None:  # noqa: ARG002
        raise NotImplementedError("E2BDeployment.delete: see create()")

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:  # noqa: ARG002
        raise NotImplementedError("E2BDeployment.get: see create()")
