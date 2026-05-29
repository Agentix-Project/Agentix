"""Tests for `agentix deploy`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentix.deployment.docker import DockerDeploymentConfig

import agentix.cli.deploy as deploy_mod
from agentix.deployment.base import MaterializedBundle


class FakeMaterializer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None, str | None]] = []

    async def materialize_bundle(
        self,
        bundle: Path,
        *,
        name: str | None = None,
        platform: str | None = None,
    ) -> MaterializedBundle:
        self.calls.append((bundle, name, platform))
        return MaterializedBundle(
            bundle=name or "demo:1.0.0",
            platform=platform,
            metadata={"cache": "/tmp/agentix-runtime-pytest"},
        )


def test_deploy_invokes_materializer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instance = FakeMaterializer()
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    monkeypatch.setattr(deploy_mod, "load_deployment", lambda name: lambda: instance)

    assert deploy_mod.main(["fake", str(bundle), "--name", "demo:dev", "--platform", "linux/amd64"]) == 0

    assert instance.calls == [(bundle, "demo:dev", "linux/amd64")]
    output = capsys.readouterr().out
    assert "bundle -> demo:dev" in output
    assert "platform -> linux/amd64" in output
    assert "cache -> /tmp/agentix-runtime-pytest" in output


def test_deploy_passes_docker_compatible_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configs: list[DockerDeploymentConfig] = []
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    class FakeDockerDeployment(FakeMaterializer):
        def __init__(self, config: DockerDeploymentConfig) -> None:
            super().__init__()
            configs.append(config)

    monkeypatch.setattr(deploy_mod, "load_deployment", lambda name: FakeDockerDeployment)

    assert deploy_mod.main(["podman", str(bundle), "--run-arg", "--runtime=crun"]) == 0

    assert configs == [
        DockerDeploymentConfig(
            container_bin="podman",
            run_args=["--runtime=crun"],
        )
    ]


def test_deploy_rejects_backend_without_materializer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeDeployment:
        pass

    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")
    monkeypatch.setattr(deploy_mod, "load_deployment", lambda name: FakeDeployment)

    with pytest.raises(SystemExit, match="cannot materialize"):
        deploy_mod.main(["fake", str(bundle)])


def test_deploy_format_json_emits_machine_readable_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instance = FakeMaterializer()
    bundle = tmp_path / "bundle.tar"
    bundle.write_text("placeholder")

    monkeypatch.setattr(deploy_mod, "load_deployment", lambda name: lambda: instance)

    assert (
        deploy_mod.main(
            ["fake", str(bundle), "--name", "demo:dev", "--platform", "linux/amd64", "--format", "json"]
        )
        == 0
    )

    data = json.loads(capsys.readouterr().out)
    assert data["bundle"] == "demo:dev"
    assert data["platform"] == "linux/amd64"
    assert data["metadata"]["cache"] == "/tmp/agentix-runtime-pytest"
