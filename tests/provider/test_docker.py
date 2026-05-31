"""Tests for the local Docker provider backend."""

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


def _deployed_bundle(tmp_path: Path) -> Path:
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
    deployed = _deployed_bundle(tmp_path)

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

    provider = DockerProvider()
    await provider.create(
        SandboxConfig(
            image="python:3.13-slim",
            bundle=str(deployed),
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
        f"type=bind,source={(deployed / 'nix').resolve()},target=/nix,readonly"
    )
    assert run_call[run_call.index("--entrypoint") + 1] == BUNDLE_RUNTIME_ENTRYPOINT
    assert BUNDLE_RUNTIME_ENTRYPOINT == "/nix/runtime/bootstrap.sh"


@pytest.mark.asyncio
async def test_create_passes_resource_limits_and_extra_runner_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    deployed = _deployed_bundle(tmp_path)

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

    provider = DockerProvider(
        DockerProviderConfig(
            run_args=["--runtime=crun", "--cgroups=disabled"],
        )
    )
    await provider.create(
        SandboxConfig(
            image="python:3.13-slim",
            bundle=str(deployed),
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
    deployed = _deployed_bundle(tmp_path)

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

    provider = DockerProvider(DockerProviderConfig(network="host"))
    await provider.create(SandboxConfig(image="python:3.13-slim", bundle=str(deployed)))

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
    deployed = _deployed_bundle(tmp_path)

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

    provider = DockerProvider()
    with pytest.raises(TimeoutError):
        await provider.create(SandboxConfig(image="python:3.13-slim", bundle=str(deployed)))

    # The started-but-unhealthy container must be removed and its port released.
    assert any(call[0] == "rm" and call[1] == "-f" for call in calls), calls
    assert provider._ports == {}


def test_allocate_port_reserves_distinct_ports() -> None:
    provider = DockerProvider()
    ports = [provider._allocate_port() for _ in range(10)]
    # Every allocation is reserved, so none repeats even though each bind
    # socket is closed before the next (which the kernel could otherwise reuse).
    assert len(set(ports)) == 10
    assert set(ports) <= provider._inflight_ports


@pytest.mark.asyncio
async def test_failed_create_releases_reserved_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployed = _deployed_bundle(tmp_path)

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

    provider = DockerProvider()
    with pytest.raises(SystemExit):
        await provider.create(SandboxConfig(image="python:3.13-slim", bundle=str(deployed)))

    # The port reserved for the failed create must be released, not leaked.
    assert provider._inflight_ports == set()
    assert provider._ports == {}


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
async def test_container_engine_config_selects_podman(monkeypatch: pytest.MonkeyPatch) -> None:
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

    await docker_mod._docker("ps", config=DockerProviderConfig(container_engine="podman"))

    assert calls == [("podman", "ps")]


@pytest.mark.asyncio
async def test_deploy_bundle_extracts_tar_to_cache(
    tmp_path: Path,
) -> None:
    bundle = _bundle_tar(tmp_path)
    cache = tmp_path / "cache"

    provider = DockerProvider(DockerProviderConfig(bundle_cache_dir=cache))
    result = await provider.deploy_bundle(bundle, name="localhost/demo:dev")

    expected = cache / f"sha256-{'a' * 64}"
    assert result.bundle == str(expected)
    assert result.platform == "linux/amd64"
    assert result.metadata["cache"] == str(expected)
    assert result.metadata["name"] == "localhost/demo:dev"
    assert (expected / "manifest.json").is_file()
    assert (expected / "nix" / "runtime" / "bootstrap.sh").is_file()
    # `agentix deploy` surfaces these in shell-comment style so the user can
    # paste them — the docker backend has no `undeploy` API, so the hint *is*
    # the cleanup path.
    assert list(result.hints) == ["inspect contents", "remove from cache"]
    assert result.hints["inspect contents"] == f"ls -la {expected}/nix/"
    assert result.hints["remove from cache"] == f"rm -rf {expected}"


@pytest.mark.asyncio
async def test_deploy_bundle_defaults_image_ref_from_manifest(
    tmp_path: Path,
) -> None:
    bundle = _bundle_tar(tmp_path)
    result = await DockerProvider(DockerProviderConfig(bundle_cache_dir=tmp_path / "cache")).deploy_bundle(
        bundle
    )

    assert result.metadata["name"] == "demo:1.0.0"


def test_podman_provider_uses_podman_by_default() -> None:
    provider = PodmanProvider()

    assert provider.config.container_engine == "podman"


def test_podman_provider_honors_explicit_config() -> None:
    """`PodmanProvider(config)` must use the caller's config verbatim,
    not silently swap in the `podman` default."""
    custom = DockerProviderConfig(container_engine="docker", run_args=["--runtime=runsc"])
    provider = PodmanProvider(custom)

    assert provider.config is custom
    assert provider.config.container_engine == "docker"


## ── deploy CLI subcommands (registered via `agentix.deploy.commands`) ──
##
## The plugin owns its own `agentix deploy <docker|podman>` subcommand.
## Tests use Click's `CliRunner` to exercise the command end-to-end with
## a monkey-patched `DockerProvider.deploy_bundle` so no real container
## engine is invoked.


def _patch_deploy_bundle(monkeypatch: pytest.MonkeyPatch, sink: list) -> None:
    """Replace `DockerProvider.deploy_bundle` with a recorder.

    Captures (config, bundle, name, platform) and returns a synthesized
    `DeployedBundle` so the subcommand's `print_deploy_result` call has
    something to render.
    """
    from agentix.provider.base import DeployedBundle

    async def fake_deploy_bundle(self, bundle, *, name=None, platform=None):
        sink.append((self.config, bundle, name, platform))
        return DeployedBundle(
            bundle="/tmp/fake-cache",
            platform=platform or "linux/amd64",
            metadata={"cache": "/tmp/fake-cache", "name": name or "demo"},
            hints={"inspect contents": "ls /tmp/fake-cache"},
        )

    monkeypatch.setattr(DockerProvider, "deploy_bundle", fake_deploy_bundle)


def test_deploy_docker_cmd_defaults_to_docker_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    sink: list = []
    _patch_deploy_bundle(monkeypatch, sink)
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    result = CliRunner().invoke(docker_mod.deploy_docker_cmd, [str(bundle)])
    assert result.exit_code == 0, result.output

    assert len(sink) == 1
    config, captured_bundle, name, platform = sink[0]
    assert config.container_engine == "docker"
    assert config.run_args == []
    assert captured_bundle == bundle
    assert name is None
    assert platform is None


def test_deploy_podman_cmd_defaults_to_podman_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    sink: list = []
    _patch_deploy_bundle(monkeypatch, sink)
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    result = CliRunner().invoke(docker_mod.deploy_podman_cmd, [str(bundle)])
    assert result.exit_code == 0, result.output

    assert sink[0][0].container_engine == "podman"


def test_deploy_docker_cmd_passes_container_engine_and_run_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    sink: list = []
    _patch_deploy_bundle(monkeypatch, sink)
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    result = CliRunner().invoke(
        docker_mod.deploy_docker_cmd,
        [
            str(bundle),
            "--container-engine",
            "nerdctl",
            "--run-arg",
            "--runtime=runsc",
            "--run-arg",
            "--cgroups=disabled",
            "--name",
            "demo:dev",
            "--platform",
            "linux/arm64",
        ],
    )
    assert result.exit_code == 0, result.output

    config, _, name, platform = sink[0]
    assert config.container_engine == "nerdctl"
    assert config.run_args == ["--runtime=runsc", "--cgroups=disabled"]
    assert name == "demo:dev"
    assert platform == "linux/arm64"


def test_deploy_docker_cmd_renders_hints_in_text_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    _patch_deploy_bundle(monkeypatch, [])
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    result = CliRunner().invoke(docker_mod.deploy_docker_cmd, [str(bundle)])
    assert result.exit_code == 0, result.output
    assert "bundle -> /tmp/fake-cache" in result.output
    assert "# inspect contents\nls /tmp/fake-cache" in result.output


def test_deploy_docker_cmd_json_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    _patch_deploy_bundle(monkeypatch, [])
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    result = CliRunner().invoke(docker_mod.deploy_docker_cmd, [str(bundle), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["bundle"] == "/tmp/fake-cache"
    assert data["hints"] == {"inspect contents": "ls /tmp/fake-cache"}
