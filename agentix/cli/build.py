"""`agentix build` — package a Python project into a runtime image.

Usage:

    agentix build                         # current directory's project
    agentix build path/to/project         # explicit project root
    agentix build . --name hello-agentix  # NAME (auto-appends :<version>)
    agentix build . --name hello:dev      # NAME:TAG (used verbatim)
    agentix build . --platform linux/amd64
    agentix build . --dry-run             # stage the build context only

The argument is a project root — a directory with `pyproject.toml` +
`uv.lock`. The build splits cleanly along one line:

  * **Python deps** are uv's job. Inside the build container `uv venv`
    + `uv sync` materialize the project's full dependency closure into
    `/nix/runtime/venv` — exactly the venv uv would produce anywhere.
  * **System deps** are Nix's job. The interpreter + uv come from a
    Nix toolchain closure; plugins and the project contribute
    `{ pkgs }: drv` files that Nix builds and `symlinkJoin`s into
    `/nix/runtime`. Nix never touches Python packaging — no uv2nix.

The host side is deliberately thin: find the project's git repo, copy
it into a build context, and hand the context to `docker buildx build`.
Every heavy step — `uv venv`, `uv sync`, `nix build` — happens inside
the container. The host needs only `agentixx`, `docker`, and `git`;
no project venv, no uv, no nix.

The platform is the sandbox runtime platform, not the host platform.
On macOS, for example, `--platform linux/amd64` builds a Linux x86_64
bundle for a remote x86 sandbox.

A project that path-depends on siblings (a uv workspace, a cookbook
example) needs those siblings in the build context — so the context is
the whole git repository, with the project addressed by its subpath.

Result image layout (mounted into a task container at `/nix`):

    /nix/store/...        the closures: interpreter, uv, system deps
    /nix/runtime/venv     the uv venv (all Python deps)
    /nix/runtime/{bin,lib,...}   symlinkJoin of every closure
"""

from __future__ import annotations

import argparse
import platform as host_platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory

from agentix.cli._resolve import REPO_ROOT, detect_python_version, read_pyproject, short_name

# Directories never copied into the build context — caches, build
# outputs, virtualenvs, VCS metadata. The context is hashed by Docker;
# keeping it lean keeps builds fast and cacheable.
_SOURCE_SKIP = frozenset({
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "result",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".direnv",
    "node_modules",
})

# Files staged verbatim from `agentix/nix/` into the build context.
_BUILDER_FILES = ("flake.nix", "flake.lock", "Dockerfile", "bundle-build.sh")

_DOCKER_TO_NIX_SYSTEM = {
    "linux/amd64": "x86_64-linux",
    "linux/arm64": "aarch64-linux",
}

_PLATFORM_ALIASES = {
    "linux/amd64": "linux/amd64",
    "linux/x86-64": "linux/amd64",
    "amd64": "linux/amd64",
    "x86-64": "linux/amd64",
    "linux/arm64": "linux/arm64",
    "linux/arm64/v8": "linux/arm64",
    "linux/aarch64": "linux/arm64",
    "arm64": "linux/arm64",
    "aarch64": "linux/arm64",
}


def _run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command; raise SystemExit on failure."""
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=capture, text=True)
    if proc.returncode != 0:
        if capture:
            sys.stderr.write(proc.stderr or "")
        raise SystemExit(proc.returncode)
    return proc


def normalize_platform(value: str) -> str:
    """Normalize a user platform into Docker's OS/arch form."""
    key = value.strip().lower().replace("_", "-")
    platform = _PLATFORM_ALIASES.get(key)
    if platform is None:
        supported = ", ".join(sorted(_DOCKER_TO_NIX_SYSTEM))
        raise SystemExit(f"--platform {value!r}: supported values are {supported}")
    return platform


def detect_default_platform(machine: str | None = None) -> str:
    """Best-effort default Docker platform for the current build host.

    Agentix builds Linux container images even when invoked from macOS,
    so only the CPU architecture is inherited from the host.
    """
    raw = (machine or host_platform.machine()).strip().lower().replace("_", "-")
    if raw in {"amd64", "x86-64"}:
        return "linux/amd64"
    if raw in {"arm64", "aarch64"}:
        return "linux/arm64"
    raise SystemExit(f"cannot auto-detect Docker platform from machine {raw!r}; pass --platform")


def nix_system_for_platform(platform: str) -> str:
    """Return the Nix system matching a normalized Docker platform."""
    platform = normalize_platform(platform)
    return _DOCKER_TO_NIX_SYSTEM[platform]


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
    """Read a builder file shipped as `agentix/nix/<name>` package data."""
    f = resources.files("agentix") / "nix" / name
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
        stage/python-version   interpreter minor, read by flake.nix
        stage/nix-system       target Nix system, read by flake.nix
        stage/closures/        empty — filled in-container by `_assemble`
    """
    platform = normalize_platform(platform)
    stage.mkdir(parents=True, exist_ok=True)

    repo_dest = stage / "repo"
    shutil.copytree(
        context_root,
        repo_dest,
        ignore=shutil.ignore_patterns(*_SOURCE_SKIP),
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


def _docker_build(stage: Path, *, name: str, tag: str, project_subpath: Path, platform: str) -> str:
    """`docker buildx build` the staged context; return the image ref.

    A bare `NAME` is also tagged `NAME:latest` for convenience.
    """
    platform = normalize_platform(platform)
    ref = f"{name}:{tag}"
    tags = ["-t", ref]
    if tag != "latest":
        tags += ["-t", f"{name}:latest"]
    _run([
        "docker",
        "buildx",
        "build",
        "--platform",
        platform,
        "--load",
        *tags,
        "--build-arg",
        f"AGENTIX_PROJECT_SUBPATH={project_subpath}",
        "--progress=plain",
        str(stage),
    ])
    return ref


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentix build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="project root with pyproject.toml + uv.lock (default: current dir)",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="image NAME or NAME:TAG. Bare NAME gets ':<pyproject-version>'; "
        "NAME:TAG is used verbatim. Default: derived from pyproject.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stage the build context to ./build/<name>/ and stop",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="target Linux container platform for the sandbox runtime "
        "(linux/amd64 or linux/arm64; default: auto-detect local CPU)",
    )
    args = parser.parse_args(argv)

    src = Path(args.path).resolve()
    if not src.is_dir():
        raise SystemExit(f"{src}: not a directory")

    pyproject = read_pyproject(src)
    if not (src / "uv.lock").is_file():
        raise SystemExit(f"{src}/uv.lock missing — run `uv lock` first")

    name, tag = parse_name(args.name, pyproject)
    python_version = detect_python_version(pyproject)
    platform = normalize_platform(args.platform) if args.platform else detect_default_platform()
    context_root, project_subpath = resolve_context(src)

    if args.dry_run:
        out = REPO_ROOT / "build" / name
        if out.exists():
            shutil.rmtree(out)
        stage_context(out, context_root=context_root, python_version=python_version, platform=platform)
        print(f"staged build context → {out}")
        print(f"  image            → {name}:{tag}")
        print(f"  platform         → {platform}")
        print(f"  nix system       → {nix_system_for_platform(platform)}")
        print(f"  python           → 3.{python_version[1:]}")
        print(f"  context root     → {context_root}")
        print(f"  project subpath  → {project_subpath}")
        return 0

    with TemporaryDirectory(prefix="agentix-build-") as tmp:
        stage = Path(tmp) / "ctx"
        stage_context(stage, context_root=context_root, python_version=python_version, platform=platform)
        ref = _docker_build(stage, name=name, tag=tag, project_subpath=project_subpath, platform=platform)
        print(f"\nimage ready → {ref}", file=sys.stderr)
        if tag != "latest":
            print(f"            → {name}:latest", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
