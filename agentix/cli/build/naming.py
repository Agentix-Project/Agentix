"""Derive bundle names, Docker tags, and tar output paths.

Three pieces of naming logic share this module:

  * `parse_name(arg, pyproject)` — translate the user's `--name` flag
    into `(name, tag)`, falling back to the project's pyproject metadata.
  * `_tar_cache_image_ref(...)` — pick a stable, side-effect-only Docker
    tag for the tar pipeline's intermediate image (so re-builds reuse
    layers without polluting the user-visible tag space).
  * `_tar_output_path(...)` and `_default_tar_name(...)` — resolve where
    the portable bundle tar should land on disk, including filename
    sanitization for arbitrary `--name` strings.

`_artifact_component` and `_docker_tag_component` are the underlying
sanitizers — Docker tag rules and filesystem-friendly rules differ
slightly, and both have to survive arbitrary user input.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from agentix.cli.build.platform import normalize_platform
from agentix.cli.build.pyproject import short_name


def _platform_slug(platform: str) -> str:
    return normalize_platform(platform).replace("/", "-")


def _artifact_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return component or "bundle"


def _docker_tag_component(value: str) -> str:
    component = _artifact_component(value).lower().strip(".-")
    if not component:
        return "bundle"
    if not re.match(r"[a-z0-9_]", component):
        component = f"_{component}"
    return component


def _tar_cache_image_ref(*, name: str, project_subpath: Path, platform: str) -> str:
    platform = normalize_platform(platform)
    digest = hashlib.sha256(f"{name}\0{project_subpath.as_posix()}\0{platform}".encode()).hexdigest()[:12]
    platform_part = _platform_slug(platform)
    max_name_len = 128 - len(platform_part) - len(digest) - 2
    name_part = _docker_tag_component(name)[:max_name_len].rstrip(".-") or "bundle"
    return f"agentix-bundle-cache:{name_part}-{platform_part}-{digest}"


def _default_tar_name(name: str, tag: str, platform: str) -> str:
    return f"{_artifact_component(name)}-{_artifact_component(tag)}-{_platform_slug(platform)}.bundle.tar"


def _tar_output_path(output: str | None, *, name: str, tag: str, platform: str) -> Path:
    default_name = _default_tar_name(name, tag, platform)
    if output is None:
        return Path("dist") / default_name

    path = Path(output)
    if output.endswith(os.sep) or path.is_dir() or path.suffix == "":
        return path / default_name
    return path


def parse_name(arg: str | None, pyproject: dict) -> tuple[str, str]:
    """Parse `--name` into `(name, tag)`.

      * None       → (short_name, pyproject version)
      * "NAME"     → ("NAME", pyproject version)
      * "NAME:TAG" → ("NAME", "TAG")
    """
    project = pyproject.get("project", {})
    version = project.get("version")
    default_tag = version if isinstance(version, str) and version else "latest"

    if arg is None:
        return short_name(pyproject), default_tag
    if ":" in arg:
        name, _, tag = arg.partition(":")
        if not name or not tag:
            raise SystemExit(f"--name {arg!r}: both sides of ':' must be non-empty")
        return name, tag
    return arg, default_tag


__all__ = [
    "_default_tar_name",
    "_tar_cache_image_ref",
    "_tar_output_path",
    "parse_name",
]
