"""`agentix build` — package a Python project into a deploy-ready sandbox image.

Usage:

    agentix build                         # current directory's pyproject
    agentix build path/to/project         # explicit project root
    agentix build . --name hello-agentix  # NAME (auto-appends :<pyproject-version>)
    agentix build . --name hello:dev      # NAME:TAG (used verbatim)
    agentix build . --dry-run             # stage build context to ./build/<name>/, no nix invoke

The single argument is a path to a directory containing `pyproject.toml`
+ `uv.lock`. uv2nix reads the lock to materialize every Python dep as a
Nix derivation; the resulting image carries `/nix/store/...-python-env`
with the full closure (interpreter, agentixx, plugins, user code, all
transitive deps) and a `/bin/agentix-server` entry point.

Plugins contribute *system* binaries through `default.nix` files shipped
alongside their Python module. We discover them with
`importlib.resources` scanning `agentix.<short>` packages — any
`default.nix` found is composed into the bundle's PATH. The user's own
project may also drop a `default.nix` at its root to declare extra
system deps (git, ffmpeg, ...).

Build shape:

  * **One Nix evaluation per build.** uv2nix takes the workspace's
    `pyproject.toml` + `uv.lock`, produces an overlay over the chosen
    Python set, and `mkVirtualEnv` materializes the venv. Plugin
    default.nix files are imported with the same `pkgs` and joined via
    `symlinkJoin` so all paths land at `/bin/*` and `/nix/store/*`.

  * **streamLayeredImage output.** The `nix-build` result is an
    executable script. `<result> | docker load` produces an image tagged
    `<name>:<tag>` locally.

No base image, no `FROM`, no `pip install` inside the build. The bundle
is a Nix closure first, then a docker tarball.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory

from agentix.cli._resolve import REPO_ROOT, read_pyproject, short_name

_SOURCE_SKIP = {
    "__pycache__", ".venv", "build", "dist", ".git",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "result",
}


def _stage_project(src: Path, dest: Path) -> None:
    """Copy the project tree into `dest`, skipping caches and build outputs."""
    dest.mkdir(parents=True)
    for item in src.iterdir():
        if item.name in _SOURCE_SKIP or item.name.endswith(".egg-info"):
            continue
        d = dest / item.name
        if item.is_dir():
            shutil.copytree(item, d, ignore=shutil.ignore_patterns(*_SOURCE_SKIP))
        else:
            shutil.copy2(item, d)


def _stage_builder(dest: Path) -> None:
    """Copy the shipped Nix builder (flake.nix, builder.nix, flake.lock) into `dest`."""
    nix_dir = resources.files("agentix") / "nix"
    dest.mkdir(parents=True)
    for fname in ("flake.nix", "builder.nix", "flake.lock"):
        src = nix_dir / fname
        if not src.is_file():
            raise SystemExit(
                f"shipped builder missing {fname!r} at {nix_dir}. "
                f"Reinstall agentixx."
            )
        (dest / fname).write_bytes(src.read_bytes())


def _discover_plugin_nix(stage_plugin_dir: Path) -> list[str]:
    """Find every `agentix.<short>/default.nix` shipped by installed wheels.

    Returns the list of nix-relative paths (one per plugin) ready to drop
    into the generated wrapper flake. Each plugin's default.nix is copied
    into `stage_plugin_dir/<short>.nix` so the flake context is self-
    contained — Nix won't follow absolute paths outside the flake root.
    """
    stage_plugin_dir.mkdir(parents=True)
    nix_paths: list[str] = []

    try:
        agentix_root = resources.files("agentix")
    except (ModuleNotFoundError, FileNotFoundError):
        return nix_paths

    for entry in agentix_root.iterdir():
        if not entry.is_dir():
            continue
        nix_file = entry / "default.nix"
        if not nix_file.is_file():
            continue
        short = entry.name
        target = stage_plugin_dir / f"{short}.nix"
        target.write_bytes(nix_file.read_bytes())
        nix_paths.append(f"./plugins/{short}.nix")
    return nix_paths


def _detect_python_version(pp: dict) -> str:
    """Return the Nixpkgs python attr suffix (e.g. `311`, `312`, `313`).

    Reads `[project].requires-python` lower bound; defaults to `311`.
    """
    req = pp.get("project", {}).get("requires-python", "")
    # Crude: pull the first `3.x` we see.
    for token in req.replace(",", " ").split():
        token = token.lstrip(">=~^! ")
        if token.startswith("3."):
            try:
                minor = int(token.split(".")[1].rstrip(".*"))
                if 10 <= minor <= 13:
                    return f"3{minor}"
            except (ValueError, IndexError):
                continue
    return "311"


def _parse_name(arg: str | None, pp: dict) -> tuple[str, str]:
    """Parse `--name` into (name, tag).

    Accepts:
      * None              → (short_name(pp), pyproject_version)
      * "NAME"            → ("NAME", pyproject_version)
      * "NAME:TAG"        → ("NAME", "TAG")
    """
    project = pp.get("project", {})
    default_version = project.get("version", "latest")
    if not isinstance(default_version, str):
        default_version = "latest"

    if arg is None:
        return short_name(pp), default_version
    if ":" in arg:
        name, _, tag = arg.partition(":")
        if not name or not tag:
            raise SystemExit(f"--name {arg!r}: both sides of `:` must be non-empty")
        return name, tag
    return arg, default_version


def _render_wrapper(*, name: str, tag: str, python_version: str, plugin_nix_paths: list[str]) -> str:
    """Substitute the wrapper.nix.tmpl with build-specific values."""
    tmpl = (resources.files("agentix") / "nix" / "wrapper.nix.tmpl").read_text()
    plugin_list = " ".join(plugin_nix_paths)
    return (
        tmpl
        .replace("@NAME@", name)
        .replace("@TAG@", tag)
        .replace("@SYSTEM@", "x86_64-linux")
        .replace("@PYTHON_VERSION@", python_version)
        .replace("@PLUGIN_NIX_LIST@", plugin_list)
    )


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess, stream stdout/stderr, raise on non-zero exit."""
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _build(stage: Path) -> str:
    """Run `nix build` and `docker load`; return the loaded image ref."""
    _run(
        ["nix", "build", ".#bundle", "-o", "result", "--print-build-logs"],
        cwd=stage,
    )

    result = stage / "result"
    if not result.is_symlink() and not result.is_file():
        raise SystemExit(f"nix build produced no result symlink at {result}")

    # streamLayeredImage's result is a script that writes the tarball to stdout.
    print("$ ./result | docker load", file=sys.stderr)
    proc = subprocess.run(
        f"{result} | docker load",
        shell=True,
        cwd=stage,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(proc.stderr)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    # Parse "Loaded image: <ref>" from docker load output.
    for line in proc.stdout.splitlines():
        if line.startswith("Loaded image:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("docker load did not print 'Loaded image:'")


def _tag_latest(loaded: str) -> str | None:
    """Also tag the loaded image as `<name>:latest` for convenience.

    Returns the alias ref on success, or None when the loaded ref isn't
    `<name>:<version>` shape (e.g. user already passed `:latest`).
    """
    if ":" not in loaded:
        return None
    name, _, tag = loaded.rpartition(":")
    if not name or tag == "latest":
        return None
    alias = f"{name}:latest"
    proc = subprocess.run(
        ["docker", "tag", loaded, alias],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return None
    return alias


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentix build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="project root with pyproject.toml + uv.lock (default: current dir)",
    )
    parser.add_argument(
        "-n", "--name", default=None,
        help="image NAME or NAME:TAG. Bare NAME gets ':<pyproject-version>' "
             "appended; NAME:TAG is used verbatim. Default: derived from pyproject.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="stage build context to ./build/<name>/; do not invoke nix",
    )
    args = parser.parse_args(argv)

    src = Path(args.path).resolve()
    if not src.is_dir():
        raise SystemExit(f"{src}: not a directory")
    pp = read_pyproject(src)

    if not (src / "uv.lock").is_file():
        raise SystemExit(f"{src}/uv.lock missing — run `uv lock` first")

    name, tag = _parse_name(args.name, pp)
    python_version = _detect_python_version(pp)

    def _stage(stage: Path) -> None:
        _stage_builder(stage / "_builder")
        _stage_project(src, stage / "project")
        plugin_paths = _discover_plugin_nix(stage / "plugins")
        wrapper = _render_wrapper(
            name=name, tag=tag,
            python_version=python_version,
            plugin_nix_paths=plugin_paths,
        )
        (stage / "flake.nix").write_text(wrapper)
        # Flake needs a git repo to track files; init one with everything staged.
        subprocess.run(
            ["git", "init", "-q"], cwd=stage,
            stdout=subprocess.DEVNULL, check=True,
        )
        subprocess.run(
            ["git", "add", "-A"], cwd=stage,
            stdout=subprocess.DEVNULL, check=True,
        )

    if args.dry_run:
        out = REPO_ROOT / "build" / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        _stage(out)
        print(f"staged build context → {out}")
        print(f"would build → {name}:{tag}")
        print(f"  python → 3.{python_version[1:]}")
        print(f"  plugin nix files → {len(list((out / 'plugins').iterdir()))}")
        return 0

    with TemporaryDirectory(prefix="agentix-build-") as tmp:
        stage = Path(tmp)
        _stage(stage)
        loaded = _build(stage)
        alias = _tag_latest(loaded)
        print(f"\nimage ready → {loaded}", file=sys.stderr)
        if alias:
            print(f"            → {alias}", file=sys.stderr)
        # Stage path is destroyed on exit; nothing else to do.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
