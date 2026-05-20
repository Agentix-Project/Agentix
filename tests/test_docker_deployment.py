"""Tests for the local Docker deployment backend."""

from __future__ import annotations

import agentix.deployment.docker as docker_mod
import pytest
from agentix.deployment.docker import DockerDeployment

from agentix.deployment.base import SandboxConfig


def test_carrier_name_includes_platform() -> None:
    amd64 = docker_mod._carrier_name("bundle:pytest", "linux/amd64")
    arm64 = docker_mod._carrier_name("bundle:pytest", "linux/arm64")

    assert amd64 != arm64
    assert docker_mod._carrier_name("bundle:pytest") == docker_mod._carrier_name("bundle:pytest")


@pytest.mark.asyncio
async def test_create_passes_platform_to_carrier_and_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str, check: bool = True) -> tuple[int, bytes, bytes]:
        calls.append(args)
        if args[0] == "inspect":
            return 1, b"", b""
        return 0, b"", b""

    async def fake_wait_healthy(self: DockerDeployment, port: int) -> None:
        return None

    monkeypatch.setattr(docker_mod, "_docker", fake_docker)
    monkeypatch.setattr(DockerDeployment, "_allocate_port", staticmethod(lambda: 18000))
    monkeypatch.setattr(DockerDeployment, "_wait_healthy", fake_wait_healthy)

    deployment = DockerDeployment()
    await deployment.create(
        SandboxConfig(
            image="python:3.13-slim",
            runtime_image="bundle:pytest",
            platform="linux/amd64",
        )
    )

    create_call = next(call for call in calls if call[0] == "create")
    run_call = next(call for call in calls if call[0] == "run")

    assert create_call[1:3] == ("--platform", "linux/amd64")
    assert run_call[1:3] == ("--platform", "linux/amd64")
