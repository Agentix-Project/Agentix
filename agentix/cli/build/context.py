"""Stage the Docker build context for `agentix build`.

The build runs inside a container; the host's job is to assemble a
context directory that the in-container script (`bundle-build.sh`) can
work against. That context is the project's git repository — copying
the whole repo is what lets in-container `uv sync` resolve path
dependencies that point outside the project directory.

This module owns:

  * `git_toplevel(path)` — find the git work-tree root containing a
    project, or None when the project is standalone.
  * `resolve_context(src)` — pick the repo root + the subpath that
    locates the project inside it.
  * `stage_context(stage, …)` — lay out the staged context directory
    that Docker buildx will see (`repo/`, builder files, marker files,
    and an empty `closures/` to be filled in-container).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from importlib import resources
from pathlib import Path

from agentix.cli.build.platform import nix_system_for_platform, normalize_platform

# Directories never copied into the build context, no matter where in
# the tree they appear — caches, virtualenvs, VCS metadata. They have
# no business in a release wheel and only inflate the Docker context
# digest, hurting cache hits.
_SKIP_ANYWHERE = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".direnv",
    "node_modules",
})

# Directories skipped ONLY at the repo root — these are conventional
# build-output names (`python -m build` / hatchling / Nix). Stripping
# them everywhere would also strip nested packages that happen to share
# the name, e.g. `agentix/cli/build/` (the package that owns this very
# file) or any user project's `agentix/<plugin>/dist/`.
_SKIP_TOP_LEVEL = frozenset({"build", "dist", "result"})


def _make_ignore(context_root: Path) -> Callable[[str, list[str]], set[str]]:
    """Build a `shutil.copytree(ignore=...)` callback.

    `shutil.ignore_patterns` matches against basenames at every depth,
    which is exactly the wrong semantics for the `build/` / `dist/` /
    `result/` entries — they're top-level conventions, not "skip any
    directory of this name anywhere in the tree".
    """
    root = str(context_root)

    def ignore(src_dir: str, names: list[str]) -> set[str]:
        skip = {n for n in names if n in _SKIP_ANYWHERE}
        if src_dir == root:
            skip.update(n for n in names if n in _SKIP_TOP_LEVEL)
        return skip

    return ignore

# Files staged verbatim from `agentix/builder/` into the build context.
_BUILDER_FILES = (
    "flake.nix",
    "flake.lock",
    "Dockerfile",
    "bundle-build.sh",
    # Shipped verbatim into the bundle as `/nix/runtime/bootstrap.sh`
    # by `bundle-build.sh`. The container entry point deployment
    # backends exec.
    "bootstrap.sh",
)


def git_toplevel(path: Path) -> Path | None:
    """The git work-tree root containing `path`, or None when `path`
    is not inside a git repository."""
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    top = proc.stdout.strip()
    return Path(top).resolve() if top else None


def resolve_context(src: Path) -> tuple[Path, Path]:
    """Return `(context_root, project_subpath)` for a project at `src`.

    The context root is the project's git repository — copying the
    whole repo is what lets in-container `uv sync` resolve path
    dependencies that point outside the project directory (`../..`,
    `../../plugins/*`). `project_subpath` locates the project within
    the staged copy.

    A project not in a git repo is its own context (`project_subpath`
    is `.`); that supports only registry/git Python deps, since there
    is nothing outside the directory to copy.
    """
    src = src.resolve()
    top = git_toplevel(src)
    if top is None:
        return src, Path(".")
    return top, src.relative_to(top)


def _shipped(name: str) -> bytes:
    """Read a builder file shipped as `agentix/builder/<name>` package data."""
    f = resources.files("agentix.builder") / name
    if not f.is_file():
        raise SystemExit(f"shipped builder file {name!r} missing — reinstall agentixx")
    return f.read_bytes()


def stage_context(
    stage: Path,
    *,
    context_root: Path,
    python_version: str,
    platform: str,
) -> None:
    """Lay out the Docker build context under `stage`.

        stage/repo/            copy of the git repo (skip-listed)
        stage/flake.nix        Nix builder (verbatim)
        stage/flake.lock       pinned nixpkgs (verbatim)
        stage/Dockerfile       build container (verbatim)
        stage/bundle-build.sh  in-container orchestration (verbatim)
        stage/bootstrap.sh     bundle runtime entry point — copied to
                               /nix/runtime/bootstrap.sh by the in-container
                               build, then exec'd by deployment backends
        stage/python-version   interpreter minor, read by flake.nix
        stage/nix-system       target Nix system, read by flake.nix
        stage/closures/        empty — filled in-container by `closures.py`
    """
    platform = normalize_platform(platform)
    stage.mkdir(parents=True, exist_ok=True)

    repo_dest = stage / "repo"
    shutil.copytree(
        context_root,
        repo_dest,
        ignore=_make_ignore(context_root),
        symlinks=True,
    )

    for name in _BUILDER_FILES:
        (stage / name).write_bytes(_shipped(name))

    (stage / "python-version").write_text(f"{python_version}\n")
    (stage / "nix-system").write_text(f"{nix_system_for_platform(platform)}\n")
    (stage / "closures").mkdir(exist_ok=True)
    # git won't track an empty dir; the flake guards on pathExists, but
    # a marker keeps the dir present in the context tarball.
    (stage / "closures" / ".keep").write_text("")


__all__ = ["git_toplevel", "resolve_context", "stage_context"]
