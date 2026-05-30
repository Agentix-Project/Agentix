"""Tests for the local Docker deployment backend."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path

import agentix.provider.docker as docker_mod
import pytest
from agentix.provider.docker import DockerProvider, DockerProviderConfig, PodmanProvider

from agentix.provider.base import SandboxConfig, SandboxResource


def _bundle_tar(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.tar"
    manifest = {
        "schema_version": 1,
        "format": "agentix-bundle",
        "name": "demo",
        "tag": "1.0.0",
        "platform": "linux/amd64",
        "digest": "sha256:" + "a" * 64,
        "runtime_env": {"PATH": "/nix/runtime/venv/bin:/nix/runtime/bin"},
        "agentix_added_env": {"AGENTIX_ADDED_PATH": "/nix/runtime/venv/bin:/nix/runtime/bin"},
    }
    with tarfile.open(path, "w") as tar:
        manifest_bytes = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for name in ("nix", "nix/runtime"):
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        bootstrap = b"#!/bin/sh\n"
        info = tarfile.TarInfo("nix/runtime/bootstrap.sh")
        info.mode = 0o755
        info.size = len(bootstrap)
        tar.addfile(info, io.BytesIO(bootstrap))
    return path


def _materialized_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle-cache" / "sha256-test"
    nix = root / "nix" / "runtime"
    nix.mkdir(parents=True)
    (nix / "bootstrap.sh").write_text("#!/bin/sh\n")
    return root


@pytest.mark.asyncio
async def test_create_passes_platform_and_bundle_mount_to_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    materialized = _materialized_bundle(tmp_path)

    async def fake_docker(
        *args: str,
        config: DockerProviderConfig | None = None,
        check: bool = True,
        retries: int = 0,
    ) -> tuple[int, bytes, bytes]:
        del config, check, retries
        calls.append(args)
        if args[0] == "inspect":
            return 1, b"", b""
        return 0, b"", b""

    async def fake_wait_healthy(self: DockerProvider, port: int) -> None:
        return None

    monkeypatch.setattr(docker_mod, "_docker", fake_docker)
    monkeypatch.setattr(DockerProvider, "_allocate_port", staticmethod(lambda: 18000))
    monkeypatch.setattr(DockerProvider, "_wait_healthy", fake_wait_healthy)

    deployment = DockerProvider()
    await deployment.create(
        SandboxConfig(
            image="python:3.13-slim",
            bundle=str(materialized),
            platform="linux/amd64",
        )
    )

    run_call = next(call for call in calls if call[0] == "run")

    assert not any(call[0] == "create" for call in calls)
    assert not any(call[0] == "start" for call in calls)
    assert run_call[1:3] == ("--platform", "linux/amd64")
    assert run_call[run_call.index("-p") + 1] == "127.0.0.1:18000:18000"
    assert "--network" not in run_call
    # The bundle's /nix/runtime/bootstrap.sh is the runtime contract,
    # so every backend reads the same entrypoint constant.
    from agentix.runtime import BUNDLE_RUNTIME_ENTRYPOINT

    assert run_call[-1] == "python:3.13-slim"
    assert "-c" not in run_call
    assert run_call[run_call.index("--mount") + 1] == (
        f"type=bind,source={(materialized / 'nix').resolve()},target=/nix,readonly"
    )
    assert run_call[run_call.index("--entrypoint") + 1] == BUNDLE_RUNTIME_ENTRYPOINT
    assert BUNDLE_RUNTIME_ENTRYPOINT == "/nix/runtime/bootstrap.sh"


@pytest.mark.asyncio
async def test_create_passes_resource_limits_and_extra_runner_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    materialized = _materialized_bundle(tmp_path)

    async def fake_docker(
        *args: str,
        config: DockerProviderConfig | None = None,
        check: bool = True,
        retries: int = 0,
    ) -> tuple[int, bytes, bytes]:
        del config, check, retries
        calls.append(args)
        if args[0] == "inspect":
            return 1, b"", b""
        return 0, b"", b""

    async def fake_wait_healthy(self: DockerProvider, port: int) -> None:
        return None

    monkeypatch.setattr(docker_mod, "_docker", fake_docker)
    monkeypatch.setattr(DockerProvider, "_allocate_port", staticmethod(lambda: 18001))
    monkeypatch.setattr(DockerProvider, "_wait_healthy", fake_wait_healthy)

    deployment = DockerProvider(
        DockerProviderConfig(
            run_args=["--runtime=crun", "--cgroups=disabled"],
        )
    )
    await deployment.create(
        SandboxConfig(
            image="python:3.13-slim",
            bundle=str(materialized),
            resource=SandboxResource(cpu=4, memory="16g", gpu=2),
        )
    )

    run_call = next(call for call in calls if call[0] == "run")

    assert not any(call[0] == "create" for call in calls)
    assert run_call[run_call.index("--cpus") + 1] == "4"
    assert run_call[run_call.index("--memory") + 1] == "16g"
    assert run_call[run_call.index("--gpus") + 1] == "2"
    assert "--runtime=crun" in run_call
    assert "--cgroups=disabled" in run_call


@pytest.mark.asyncio
async def test_host_network_binds_runtime_to_loopback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    materialized = _materialized_bundle(tmp_path)

    async def fake_docker(
        *args: str,
        config: DockerProviderConfig | None = None,
        check: bool = True,
        retries: int = 0,
    ) -> tuple[int, bytes, bytes]:
        del config, check, retries
        calls.append(args)
        if args[0] == "inspect":
            return 1, b"", b""
        return 0, b"", b""

    async def fake_wait_healthy(self: DockerProvider, port: int) -> None:
        return None

    monkeypatch.setattr(docker_mod, "_docker", fake_docker)
    monkeypatch.setattr(DockerProvider, "_allocate_port", staticmethod(lambda: 18004))
    monkeypatch.setattr(DockerProvider, "_wait_healthy", fake_wait_healthy)

    deployment = DockerProvider(DockerProviderConfig(network="host"))
    await deployment.create(SandboxConfig(image="python:3.13-slim", bundle=str(materialized)))

    run_call = next(call for call in calls if call[0] == "run")
    assert "--network" in run_call
    assert run_call[run_call.index("--network") + 1] == "host"
    assert "-p" not in run_call
    assert "-e" in run_call
    assert "AGENTIX_BIND_HOST=127.0.0.1" in run_call


@pytest.mark.asyncio
async def test_create_removes_container_when_health_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    materialized = _materialized_bundle(tmp_path)

    async def fake_docker(
        *args: str,
        config: DockerProviderConfig | None = None,
        check: bool = True,
        retries: int = 0,
    ) -> tuple[int, bytes, bytes]:
        del config, check, retries
        calls.append(args)
        return 0, b"", b""

    async def failing_wait(self: DockerProvider, port: int) -> None:
        raise TimeoutError("runtime never became healthy")

    monkeypatch.setattr(docker_mod, "_docker", fake_docker)
    monkeypatch.setattr(DockerProvider, "_allocate_port", staticmethod(lambda: 18010))
    monkeypatch.setattr(DockerProvider, "_wait_healthy", failing_wait)

    deployment = DockerProvider()
    with pytest.raises(TimeoutError):
        await deployment.create(SandboxConfig(image="python:3.13-slim", bundle=str(materialized)))

    # The started-but-unhealthy container must be removed and its port released.
    assert any(call[0] == "rm" and call[1] == "-f" for call in calls), calls
    assert deployment._ports == {}


def test_allocate_port_reserves_distinct_ports() -> None:
    deployment = DockerProvider()
    ports = [deployment._allocate_port() for _ in range(10)]
    # Every allocation is reserved, so none repeats even though each bind
    # socket is closed before the next (which the kernel could otherwise reuse).
    assert len(set(ports)) == 10
    assert set(ports) <= deployment._inflight_ports


@pytest.mark.asyncio
async def test_failed_create_releases_reserved_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = _materialized_bundle(tmp_path)

    async def fake_docker(
        *args: str,
        config: DockerProviderConfig | None = None,
        check: bool = True,
        retries: int = 0,
    ) -> tuple[int, bytes, bytes]:
        del config, check, retries
        if args[0] == "run":
            raise SystemExit("docker run failed")
        return 0, b"", b""

    monkeypatch.setattr(docker_mod, "_docker", fake_docker)

    deployment = DockerProvider()
    with pytest.raises(SystemExit):
        await deployment.create(SandboxConfig(image="python:3.13-slim", bundle=str(materialized)))

    # The port reserved for the failed create must be released, not leaked.
    assert deployment._inflight_ports == set()
    assert deployment._ports == {}


def test_gpu_args_can_be_overridden_for_podman_cdi() -> None:
    config = DockerProviderConfig(gpu_args=["--device", "nvidia.com/gpu=all", "--label", "gpu-count={gpu}"])

    assert docker_mod._resource_args(SandboxResource(gpu=2), config) == [
        "--device",
        "nvidia.com/gpu=all",
        "--label",
        "gpu-count=2",
    ]


def test_publish_host_can_be_omitted_for_podman_cni() -> None:
    assert docker_mod._port_mapping(18002) == "127.0.0.1:18002:18002"

    assert docker_mod._port_mapping(18002, DockerProviderConfig(publish_host="")) == "18002:18002"


def test_network_mode_disables_port_publishing() -> None:
    assert docker_mod._network_args() == []
    assert docker_mod._publish_args(18003) == ["-p", "127.0.0.1:18003:18003"]

    slirp = DockerProviderConfig(network="slirp4netns")
    assert docker_mod._network_args(slirp) == ["--network", "slirp4netns"]
    assert docker_mod._publish_args(18003, slirp) == ["-p", "127.0.0.1:18003:18003"]

    host = DockerProviderConfig(network="host")
    assert docker_mod._network_args(host) == ["--network", "host"]
    assert docker_mod._publish_args(18003, host) == []


@pytest.mark.asyncio
async def test_container_bin_config_selects_podman(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    class FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def fake_exec(*args: str, **kwargs: object) -> FakeProc:
        calls.append(args)
        assert kwargs["stdout"] == asyncio.subprocess.PIPE
        assert kwargs["stderr"] == asyncio.subprocess.PIPE
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    await docker_mod._docker("ps", config=DockerProviderConfig(container_bin="podman"))

    assert calls == [("podman", "ps")]


@pytest.mark.asyncio
async def test_materialize_bundle_extracts_tar_to_cache(
    tmp_path: Path,
) -> None:
    bundle = _bundle_tar(tmp_path)
    cache = tmp_path / "cache"

    deployment = DockerProvider(DockerProviderConfig(bundle_cache_dir=cache))
    result = await deployment.materialize_bundle(bundle, name="localhost/demo:dev")

    expected = cache / f"sha256-{'a' * 64}"
    assert result.bundle == str(expected)
    assert result.platform == "linux/amd64"
    assert result.metadata["cache"] == str(expected)
    assert result.metadata["name"] == "localhost/demo:dev"
    assert (expected / "manifest.json").is_file()
    assert (expected / "nix" / "runtime" / "bootstrap.sh").is_file()


@pytest.mark.asyncio
async def test_materialize_bundle_defaults_image_ref_from_manifest(
    tmp_path: Path,
) -> None:
    bundle = _bundle_tar(tmp_path)
    result = await DockerProvider(DockerProviderConfig(bundle_cache_dir=tmp_path / "cache")).materialize_bundle(
        bundle
    )

    assert result.metadata["name"] == "demo:1.0.0"


def test_podman_deployment_uses_podman_by_default() -> None:
    deployment = PodmanProvider()

    assert deployment._deployment.config.container_bin == "podman"
