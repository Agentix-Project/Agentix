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

    async def fake_docker(*args: str, check: bool = True, retries: int = 0) -> tuple[int, bytes, bytes]:
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
            bundle="bundle:pytest",
            platform="linux/amd64",
        )
    )

    create_call = next(call for call in calls if call[0] == "create")
    run_call = next(call for call in calls if call[0] == "run")

    assert create_call[1:3] == ("--platform", "linux/amd64")
    assert run_call[1:3] == ("--platform", "linux/amd64")
    assert run_call[run_call.index("-p") + 1] == "127.0.0.1:18000:18000"
    assert "--network" not in run_call
    # The bundle's own /nix/runtime/bootstrap.sh is the container
    # entrypoint — no more `-c '<inline bootstrap>'` indirection. The
    # script ships with the bundle (built by
    # `agentix/builder/bundle-build.sh` from `agentix/builder/bootstrap.sh`).
    # Path lives on `agentix.deployment.base.BUNDLE_RUNTIME_ENTRYPOINT`
    # so every backend reads the same constant.
    from agentix.deployment.base import BUNDLE_RUNTIME_ENTRYPOINT

    assert run_call[-1] == "python:3.13-slim"
    assert "-c" not in run_call
    assert (
        run_call[run_call.index("--entrypoint") + 1]
        == BUNDLE_RUNTIME_ENTRYPOINT
    )
    assert BUNDLE_RUNTIME_ENTRYPOINT == "/nix/runtime/bootstrap.sh"


@pytest.mark.asyncio
async def test_carrier_recreated_when_runtime_tag_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    carrier = docker_mod._carrier_name("bundle:pytest", "linux/amd64")
    calls: list[tuple[str, ...]] = []

    async def fake_docker(*args: str, check: bool = True, retries: int = 0) -> tuple[int, bytes, bytes]:
        calls.append(args)
        if args == ("inspect", carrier):
            return 0, b"", b""
        if args == ("inspect", "-f", "{{.Image}}", carrier):
            return 0, b"sha256:old\n", b""
        if args == ("image", "inspect", "-f", "{{.Id}}", "bundle:pytest"):
            return 0, b"sha256:new\n", b""
        return 0, b"", b""

    monkeypatch.setattr(docker_mod, "_docker", fake_docker)

    deployment = DockerDeployment()
    assert await deployment._ensure_carrier("bundle:pytest", "linux/amd64") == carrier

    rm_call = calls.index(("rm", "-f", carrier))
    create_call = calls.index(("create", "--platform", "linux/amd64", "--name", carrier, "bundle:pytest"))
    assert rm_call < create_call
