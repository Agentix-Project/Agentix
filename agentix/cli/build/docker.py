"""Invoke `docker buildx` against a staged build context.

This module is intentionally narrow: a single subprocess helper that
echoes commands and surfaces failures as `SystemExit`, plus two
wrappers around `docker buildx build --load`. The first is the
low-level form used by the tar pipeline (one cache tag, no auto
`:latest`); the second is the user-facing wrapper for `--format
oci-image` that mirrors the bare-NAME → `NAME:latest` convenience.

Heavy lifting — `uv sync`, `nix build` — happens inside the container
once buildx kicks off; the host never sees Python or Nix directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agentix.cli.build.platform import normalize_platform


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command; raise SystemExit on failure."""
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True)
    if check and proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stderr or "")
        raise SystemExit(proc.returncode)
    return proc


def _docker_build_image(stage: Path, *, tags: list[str], project_subpath: Path, platform: str) -> None:
    """`docker buildx build --load` the staged context with explicit tags."""
    if not tags:
        raise SystemExit("internal error: docker image build requires at least one tag")
    _run([
        "docker",
        "buildx",
        "build",
        "--platform",
        normalize_platform(platform),
        "--load",
        *(arg for tag in tags for arg in ("-t", tag)),
        "--build-arg",
        f"AGENTIX_PROJECT_SUBPATH={project_subpath}",
        "--progress=plain",
        str(stage),
    ])


def _docker_build(stage: Path, *, name: str, tag: str, project_subpath: Path, platform: str) -> str:
    """Build the Docker-compatible bundle image; return the primary image ref.

    A bare `NAME` is also tagged `NAME:latest` for convenience.
    """
    ref = f"{name}:{tag}"
    tags = [ref]
    if tag != "latest":
        tags.append(f"{name}:latest")
    _docker_build_image(stage, tags=tags, project_subpath=project_subpath, platform=platform)
    return ref


__all__ = ["_docker_build", "_docker_build_image", "_run"]
