"""`SandboxProvider.session()` teardown contract: the sandbox is deleted even
when closing its runtime client fails — otherwise the container (and its
reserved port) leaks on every aclose error."""

from __future__ import annotations

import pytest

from agentix.provider.base import Sandbox, SandboxConfig, SandboxId, SandboxInfo, SandboxProvider


class _ExplodingCloseSandbox(Sandbox):
    async def aclose(self) -> None:  # type: ignore[override]
        raise RuntimeError("close boom")


class _Provider(SandboxProvider):
    def __init__(self) -> None:
        self.deleted: list[SandboxId] = []

    async def create(self, config: SandboxConfig) -> Sandbox:
        return _ExplodingCloseSandbox(
            sandbox_id=SandboxId("s-1"), runtime_url="http://127.0.0.1:1", status="running"
        )

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        return SandboxInfo(sandbox_id=sandbox_id, runtime_url="http://127.0.0.1:1", status="running")

    async def delete(self, sandbox_id: SandboxId) -> None:
        self.deleted.append(sandbox_id)


async def test_session_deletes_sandbox_even_when_aclose_fails() -> None:
    provider = _Provider()
    with pytest.raises(RuntimeError, match="close boom"):
        async with provider.session(SandboxConfig(image="x", bundle="y")):
            pass
    assert provider.deleted == [SandboxId("s-1")]
