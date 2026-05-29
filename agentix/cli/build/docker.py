"""Invoke a container build CLI against a staged build context.

This module is intentionally narrow: a single subprocess helper that
echoes commands and surfaces failures as `SystemExit`, plus the image
build helper used by the tar pipeline. Docker remains the default and
uses `docker buildx build --load`; Podman can be selected by passing
`ContainerBuildConfig(container_bin="podman", ...)`.

Heavy lifting — `uv sync`, `nix build` — happens inside the container
once buildx kicks off; the host never sees Python or Nix directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agentix.cli.build.platform import normalize_platform

# Host env vars that, when present, are forwarded into the build container
# verbatim as `--build-arg KEY=VALUE`. They're declared as `ARG` in the
# Dockerfile and propagated to subsequent `RUN` steps via matching `ENV`.
#
#   NIX_CONFIG            Nix's official `nix.conf` override hatch — covers
#                         substituters, trusted-public-keys, timeouts, and
#                         every other setting in one knob.
#   AGENTIX_BUILDER_BASE  Builder base image, used directly in `FROM`.
#
# HTTP proxies are intentionally not in this list: Docker BuildKit reads
# them from `~/.docker/config.json`'s `proxies` section, which is the
# durable, daemon-level place to configure them.
_ENV_BUILD_ARG_NAMES = ("NIX_CONFIG", "AGENTIX_BUILDER_BASE")


@dataclass(frozen=True)
class ContainerBuildConfig:
    """Docker-compatible build executor settings."""

    container_bin: str = "docker"
    container_args: tuple[str, ...] = ()
    container_run_args: tuple[str, ...] = ()


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command; raise SystemExit on failure."""
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"container build CLI {cmd[0]!r} not found on PATH. Install Docker "
            f"(https://docs.docker.com/get-docker/) or Podman "
            f"(https://podman.io/docs/installation), or pass --container-bin."
        ) from exc
    if check and proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stderr or "")
        raise SystemExit(proc.returncode)
    return proc


def _build_container_bin(config: ContainerBuildConfig | None = None) -> str:
    return (config or ContainerBuildConfig()).container_bin


def _build_container_args(config: ContainerBuildConfig | None = None) -> list[str]:
    return list((config or ContainerBuildConfig()).container_args)


def _build_container_run_args(config: ContainerBuildConfig | None = None) -> list[str]:
    return list((config or ContainerBuildConfig()).container_run_args)


def _env_build_args() -> list[str]:
    """Forward known host env vars into the build as `--build-arg KEY=VALUE`."""
    return [
        arg
        for name in _ENV_BUILD_ARG_NAMES
        if (value := os.environ.get(name))
        for arg in ("--build-arg", f"{name}={value}")
    ]


def _docker_build_image(
    stage: Path,
    *,
    tags: list[str],
    project_subpath: Path,
    platform: str,
    config: ContainerBuildConfig | None = None,
) -> None:
    """Build the staged context with explicit tags."""
    if not tags:
        raise SystemExit("internal error: container image build requires at least one tag")
    config = config or ContainerBuildConfig()
    bin_name = config.container_bin
    tags_args = [arg for tag in tags for arg in ("-t", tag)]
    if bin_name == "docker":
        cmd = [
            bin_name,
            "buildx",
            "build",
            "--platform",
            normalize_platform(platform),
            "--load",
            *_build_container_args(config),
            *_env_build_args(),
            *tags_args,
            "--build-arg",
            f"AGENTIX_PROJECT_SUBPATH={project_subpath}",
            "--progress=plain",
            str(stage),
        ]
    else:
        cmd = [
            bin_name,
            "build",
            "--platform",
            normalize_platform(platform),
            *_build_container_args(config),
            *_env_build_args(),
            *tags_args,
            "--build-arg",
            f"AGENTIX_PROJECT_SUBPATH={project_subpath}",
            str(stage),
        ]
    _run(cmd)


__all__ = [
    "_build_container_bin",
    "_build_container_run_args",
    "ContainerBuildConfig",
    "_docker_build_image",
    "_run",
]
