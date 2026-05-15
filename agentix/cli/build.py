"""`agentix build` — build a closure image from `primitives/<name>/`.

Usage:

    agentix build primitives/bash
    agentix build primitives/files --tag agentix/primitive-files:dev
    agentix build primitives/bash --dry-run    # stage context, no docker

A closure's source directory is the minimum-viable layout:

    primitives/<name>/
    ├── pyproject.toml                # all metadata (name, version, description)
    └── agentix_closures/<name>/
        ├── __init__.py               # stub class
        └── _impl.py                  # impl class

Everything else — Dockerfile, default.nix, manifest.json — is shared
infrastructure pulled in from `primitives/_template/` and `tools/`.
This script stages a self-contained build context in a temp dir, copies
the templates next to the closure source, and runs `docker build`. The
closure dir never has to carry build-time boilerplate.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

# `agentix.cli.build` lives inside the installed framework; the templates +
# gen_manifest tool live alongside the repo root. Walk up two levels from
# this file (agentix/cli/build.py → repo) to find them when running from a
# dev checkout. In an installed wheel these files don't ship, so
# `agentix build` is a dev-time tool by design.
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "primitives" / "_template"
GEN_MANIFEST = REPO_ROOT / "tools" / "gen_manifest.py"


def _load_pyproject(closure_dir: Path) -> dict:
    pp = closure_dir / "pyproject.toml"
    if not pp.is_file():
        raise SystemExit(f"{closure_dir}: missing pyproject.toml")
    with pp.open("rb") as f:
        return tomllib.load(f)


def _derive_tag(pyproject: dict) -> str:
    """`agentix-primitive-bash` v `0.1.0` → `agentix/primitive-bash:0.1.0`.

    The framework's convention is `agentix-<kind>-<name>` for the
    distribution and `agentix/<kind>-<name>:<version>` for the image.
    """
    project = pyproject.get("project", {})
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise SystemExit("pyproject.toml: [project] needs string name + version")
    if not name.startswith("agentix-"):
        raise SystemExit(
            f"pyproject.toml: name {name!r} must follow `agentix-<kind>-<name>`"
        )
    short = name[len("agentix-"):]  # e.g. primitive-bash
    return f"agentix/{short}:{version}"


def _stage(closure_dir: Path, build_dir: Path) -> None:
    """Copy closure source + shared build infra into a self-contained context."""
    shutil.copytree(
        closure_dir / "agentix_closures",
        build_dir / "agentix_closures",
    )
    shutil.copy2(closure_dir / "pyproject.toml", build_dir / "pyproject.toml")
    shutil.copy2(TEMPLATE_DIR / "Dockerfile", build_dir / "Dockerfile")
    shutil.copy2(TEMPLATE_DIR / "default.nix", build_dir / "default.nix")
    shutil.copy2(GEN_MANIFEST, build_dir / "gen_manifest.py")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentix build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("closure_dir", type=Path, help="path to primitives/<name>/")
    parser.add_argument("--tag", type=str, default=None,
                        help="override the derived docker image tag")
    parser.add_argument("--dry-run", action="store_true",
                        help="stage the build context to ./build/<name>/ and print path; "
                             "do NOT invoke docker")
    args = parser.parse_args(argv)

    if not args.closure_dir.is_dir():
        raise SystemExit(f"{args.closure_dir}: not a directory")
    pyproject = _load_pyproject(args.closure_dir)
    tag = args.tag or _derive_tag(pyproject)
    short_name = pyproject["project"]["name"].rsplit("-", 1)[-1]

    if args.dry_run:
        out = REPO_ROOT / "build" / short_name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        _stage(args.closure_dir, out)
        print(f"staged build context → {out}")
        print(f"would build → {tag}")
        return 0

    with TemporaryDirectory(prefix=f"agentix-build-{short_name}-") as tmp:
        build_dir = Path(tmp)
        _stage(args.closure_dir, build_dir)
        print(f"building {tag} from {args.closure_dir}…", file=sys.stderr)
        proc = subprocess.run(
            [
                "docker", "build",
                "--build-arg", f"CLOSURE_NAME={short_name}",
                "-t", tag,
                str(build_dir),
            ],
            check=False,
        )
        return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
