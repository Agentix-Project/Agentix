"""E2B provider backend — stub.

E2B (https://e2b.dev/) hosts ephemeral sandboxes seeded by their own
template and filesystem mechanisms. A bundle tar needs to be uploaded
or unpacked into an E2B-accessible volume before it can be mounted at
`/nix`.
The class exists so `agentix deploy e2b` fails with a clear error
and so callers can write code against the real Protocol contract.

Config comes from env: `E2B_API_KEY` and `E2B_TEMPLATE_ID`. No
constructor arguments — plugin loaders instantiate this with `cls()`.
"""

from __future__ import annotations

import os

from agentix.provider.base import Sandbox, SandboxConfig, SandboxId, SandboxInfo, SandboxProvider


class E2BProvider(SandboxProvider):
    """Sandbox CRUD via E2B (pending integration)."""

    def __init__(self) -> None:
        self._api_key = os.environ.get("E2B_API_KEY")
        self._template_id = os.environ.get("E2B_TEMPLATE_ID")

    async def create(self, config: SandboxConfig) -> Sandbox:  # noqa: ARG002
        raise NotImplementedError(
            "E2BProvider is not wired yet. A bundle tar needs to be "
            "uploaded or unpacked into an E2B-accessible volume before "
            "sandbox creation; the API integration is on the deploy roadmap."
        )

    async def delete(self, sandbox_id: SandboxId) -> None:  # noqa: ARG002
        raise NotImplementedError("E2BProvider.delete: see create()")

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:  # noqa: ARG002
        raise NotImplementedError("E2BProvider.get: see create()")
