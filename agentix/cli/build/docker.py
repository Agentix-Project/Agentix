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
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from agentix.cli.build.platform import normalize_platform

_PROXY_BUILD_ARG_NAMES = ("http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")


@dataclass(frozen=True)
class ContainerBuildConfig:
    """Docker-compatible build executor settings + raw engine passthrough args.

    `nix_args` / `uv_args` are forwarded verbatim into the in-container
    `nix build` / `uv sync`; `container_args` / `container_run_args` are
    forwarded to the container build / export `run`. This raw passthrough
    is the escape hatch that keeps the CLI from sprouting a bespoke flag
    per engine knob.
    """

    container_bin: str = "docker"
    container_args: tuple[str, ...] = ()
    container_run_args: tuple[str, ...] = ()
    nix_args: tuple[str, ...] = ()
    uv_args: tuple[str, ...] = ()


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
            f"(https://podman.io/docs/installation), or pass --backend."
        ) from exc
    if check and proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stderr or "")
        raise SystemExit(proc.returncode)
    return proc


def _build_container_bin(config: ContainerBuildConfig | None = None) -> str:
    return (config or ContainerBuildConfig()).container_bin


def _split_passthrough(values: tuple[str, ...]) -> list[str]:
    """Shlex-split each passthrough value so a whole shell-style sub-flag can be
    given as one quoted arg, e.g. --container-arg '--build-arg FOO=bar'."""
    return [token for value in values for token in shlex.split(value)]


def _build_container_args(config: ContainerBuildConfig | None = None) -> list[str]:
    return _split_passthrough((config or ContainerBuildConfig()).container_args)


def _build_container_run_args(config: ContainerBuildConfig | None = None) -> list[str]:
    return _split_passthrough((config or ContainerBuildConfig()).container_run_args)


def _proxy_build_args() -> list[str]:
    return [
        arg
        for name in _PROXY_BUILD_ARG_NAMES
        if (value := os.environ.get(name))
        for arg in ("--build-arg", f"{name}={value}")
    ]


def _passthrough_build_args(config: ContainerBuildConfig | None = None) -> list[str]:
    """Pack raw nix/uv passthrough args into Docker build-args the in-container
    `bundle-build.sh` splits back onto `nix build` / `uv sync`."""
    config = config or ContainerBuildConfig()
    args: list[str] = []
    if config.nix_args:
        args.extend(["--build-arg", f"AGENTIX_NIX_ARGS={' '.join(config.nix_args)}"])
    if config.uv_args:
        args.extend(["--build-arg", f"AGENTIX_UV_ARGS={' '.join(config.uv_args)}"])
    return args


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
            *_proxy_build_args(),
            *_passthrough_build_args(config),
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
            *_proxy_build_args(),
            *_passthrough_build_args(config),
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
