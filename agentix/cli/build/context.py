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


def _skip_relpath(rel: str) -> bool:
    """Whether a repo-relative path should be skipped — the same skip-list as
    `_make_ignore`, applied to a `git ls-files` path: `_SKIP_ANYWHERE` names at
    any depth, `_SKIP_TOP_LEVEL` names only at the repo root."""
    parts = rel.split("/")
    if any(part in _SKIP_ANYWHERE for part in parts):
        return True
    return bool(parts) and parts[0] in _SKIP_TOP_LEVEL


def _git_listed_files(context_root: Path) -> list[str] | None:
    """Repo-relative paths git would include: tracked files plus
    untracked-but-not-ignored ones (`--exclude-standard` honors `.gitignore`,
    `.git/info/exclude`, and the global excludes). Returns None when
    `context_root` is not a git work tree."""
    proc = subprocess.run(
        ["git", "-C", str(context_root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    # Read bytes and split on NUL: `text=True` would translate newlines and
    # mangle a path containing a CR, defeating the `-z` delimiting.
    return [p.decode() for p in proc.stdout.split(b"\0") if p]


def _copy_repo(context_root: Path, repo_dest: Path) -> None:
    """Stage the project's repo into `repo_dest`.

    Prefer the set of files git tracks or leaves untracked-but-not-ignored, so
    `.gitignore`'d content — caches, virtualenvs, `.env`/secrets, large local
    data — never enters the build context or the image layers built from it.
    Falls back to a filtered working-tree copy when the project isn't in a git
    repo. The `_SKIP_*` list still applies as a backstop for un-ignored caches.
    """
    listed = _git_listed_files(context_root)
    if listed is None:
        shutil.copytree(context_root, repo_dest, ignore=_make_ignore(context_root), symlinks=True)
        return
    # Always create the destination, even if the listing is empty — the build
    # context's `repo/` directory must exist for the in-container `COPY`.
    repo_dest.mkdir(parents=True, exist_ok=True)
    for rel in listed:
        if _skip_relpath(rel):
            continue
        src = context_root / rel
        if not src.is_symlink() and not src.exists():
            continue  # listed but removed from the working tree
        dest = repo_dest / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest, follow_symlinks=False)

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
    _copy_repo(context_root, repo_dest)

    for name in _BUILDER_FILES:
        (stage / name).write_bytes(_shipped(name))

    (stage / "python-version").write_text(f"{python_version}\n")
    (stage / "nix-system").write_text(f"{nix_system_for_platform(platform)}\n")
    (stage / "closures").mkdir(exist_ok=True)
    # git won't track an empty dir; the flake guards on pathExists, but
    # a marker keeps the dir present in the context tarball.
    (stage / "closures" / ".keep").write_text("")


__all__ = ["git_toplevel", "resolve_context", "stage_context"]
