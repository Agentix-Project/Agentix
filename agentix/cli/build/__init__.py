"""`agentix build` — package a Python project into a bundle artifact.

This module is the click command itself. The pipeline pieces it
orchestrates live in sibling modules so each one stays focused on a
single concern:

  * `pyproject` — read project metadata from `pyproject.toml`.
  * `platform`  — normalize Docker/Nix platform strings.
  * `context`   — find the git repo and stage the build context.
  * `docker`    — invoke `docker buildx build`.
  * `naming`    — derive bundle name/tag and output paths.
  * `bundle`    — stream the image tar back out and assemble a portable
                  `manifest.json + nix/` archive.
  * `closures`  — in-container Nix-closure discovery (entry point for
                  `python -m agentix.cli.build.closures`, not the host
                  build pipeline).

The user-visible CLI surface is the `build` click command exported
here; `cli/__init__.py` registers it on the top-level `agentix` group
so `agentix build [PATH]` works as a subcommand.

`agentix build` takes one project root — a directory with
`pyproject.toml` + `uv.lock`. The build splits cleanly along one line:

  * Python deps are uv's job. Inside the build container `uv venv` +
    `uv sync` materialize the project's full dependency closure into
    `/nix/runtime/venv` — exactly the venv uv would produce anywhere.
  * System deps are Nix's job. The interpreter + uv come from a Nix
    toolchain closure; plugins and the project contribute
    `{ pkgs }: drv` files that Nix builds and `symlinkJoin`s into
    `/nix/runtime`. Nix never touches Python packaging — no uv2nix.

The host side is deliberately thin: find the project's git repo, copy
it into a build context, and hand the context to a Docker build
executor. Every heavy step — `uv venv`, `uv sync`, `nix build` —
happens inside the container.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from agentix.cli.build.bundle import _build_tar_bundle
from agentix.cli.build.context import resolve_context, stage_context
from agentix.cli.build.docker import _docker_build
from agentix.cli.build.naming import _tar_output_path, parse_name
from agentix.cli.build.platform import (
    detect_default_platform,
    nix_system_for_platform,
    normalize_platform,
)
from agentix.cli.build.pyproject import REPO_ROOT, detect_python_version, read_pyproject

_BUILD_FORMATS = ("tar", "oci-image")

# Click's default help formatter rewraps each paragraph, which would
# mangle the indented examples and the bundle-layout tree below. A `\b`
# character at the start of a paragraph tells click to leave its
# whitespace alone — that's how we keep the manual page rendering well.
_BUILD_HELP = """\
Package a Python project into a bundle artifact.

The argument is a project root — a directory with `pyproject.toml` +
`uv.lock`. The build splits cleanly along one line:

\b
  - Python deps are uv's job. Inside the build container `uv venv` +
    `uv sync` materialize the project's full dependency closure into
    `/nix/runtime/venv` — exactly the venv uv would produce anywhere.
  - System deps are Nix's job. The interpreter + uv come from a Nix
    toolchain closure; plugins and the project contribute
    `{ pkgs }: drv` files that Nix builds and `symlinkJoin`s into
    `/nix/runtime`. Nix never touches Python packaging — no uv2nix.

The host side is deliberately thin: find the project's git repo, copy it
into a build context, and hand the context to a Docker build executor.
Every heavy step — `uv venv`, `uv sync`, `nix build` — happens inside the
container. The host needs only `agentixx`, `docker`, and `git`; no
project venv, no uv, no nix.

The platform is the sandbox runtime platform, not the host platform. On
macOS, for example, `--platform linux/amd64` builds a Linux x86_64 bundle
for a remote x86 sandbox.

\b
Examples:
    agentix build                         # current directory's project
    agentix build path/to/project         # explicit project root
    agentix build . --name hello-agentix  # bundle tar (auto-appends version)
    agentix build . --name hello:dev      # bundle tar tagged as dev
    agentix build . --platform linux/amd64
    agentix build . --format oci-image    # Docker-compatible image path
    agentix build . --format tar          # Agentix bundle tar (default)
    agentix build . --dry-run             # stage the build context only

\b
Portable bundle tar layout:
    manifest.json                bundle identity + runtime contract
    nix/store/...                the closures: interpreter, uv, system deps
    nix/runtime/venv             the uv venv (all Python deps)
    nix/runtime/{bin,lib,...}    symlinkJoin of every closure
"""


@click.command(
    name="build",
    help=_BUILD_HELP,
    short_help="Package a Python project into a bundle artifact.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("path", type=click.Path(path_type=Path), default=".", metavar="[PATH]")
@click.option(
    "-n",
    "--name",
    default=None,
    metavar="NAME[:TAG]",
    help=(
        "Bundle NAME or NAME:TAG. Bare NAME gets ':<pyproject-version>'; "
        "NAME:TAG is used verbatim. Default: derived from pyproject."
    ),
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(_BUILD_FORMATS),
    default="tar",
    show_default=True,
    help="Artifact format: 'tar' writes manifest.json + nix/; 'oci-image' loads a Docker-compatible image.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    metavar="PATH",
    help="Output file or directory for --format tar. Default: dist/<name>-<tag>-<platform>.bundle.tar",
)
@click.option(
    "--platform",
    default=None,
    metavar="PLATFORM",
    help=(
        "Target Linux container platform for the sandbox runtime "
        "(linux/amd64 or linux/arm64; default: auto-detect local CPU)."
    ),
)
@click.option("--dry-run", is_flag=True, help="Stage the build context to ./build/<name>/ and stop.")
def build(
    path: Path,
    name: str | None,
    fmt: str,
    output: str | None,
    platform: str | None,
    dry_run: bool,
) -> int:
    """Package a Python project into a bundle artifact."""
    if output and fmt != "tar":
        raise SystemExit("--output is only supported with --format tar")

    src = path.resolve()
    if not src.is_dir():
        raise SystemExit(f"{src}: not a directory")

    pyproject = read_pyproject(src)
    if not (src / "uv.lock").is_file():
        raise SystemExit(f"{src}/uv.lock missing — run `uv lock` first")

    name, tag = parse_name(name, pyproject)
    python_version = detect_python_version(pyproject)
    platform = normalize_platform(platform) if platform else detect_default_platform()
    context_root, project_subpath = resolve_context(src)
    tar_output = _tar_output_path(output, name=name, tag=tag, platform=platform)

    if dry_run:
        out = REPO_ROOT / "build" / name
        if out.exists():
            shutil.rmtree(out)
        stage_context(out, context_root=context_root, python_version=python_version, platform=platform)
        print(f"staged build context → {out}")
        print(f"  bundle           → {name}:{tag}")
        print(f"  format           → {fmt}")
        if fmt == "tar":
            print(f"  output           → {tar_output}")
        print(f"  platform         → {platform}")
        print(f"  nix system       → {nix_system_for_platform(platform)}")
        print(f"  python           → 3.{python_version[1:]}")
        print(f"  context root     → {context_root}")
        print(f"  project subpath  → {project_subpath}")
        return 0

    with TemporaryDirectory(prefix="agentix-build-") as tmp:
        stage = Path(tmp) / "ctx"
        stage_context(stage, context_root=context_root, python_version=python_version, platform=platform)
        if fmt == "oci-image":
            ref = _docker_build(stage, name=name, tag=tag, project_subpath=project_subpath, platform=platform)
            print(f"\nimage ready → {ref}", file=sys.stderr)
            if tag != "latest":
                print(f"            → {name}:latest", file=sys.stderr)
        else:
            artifact = _build_tar_bundle(
                stage,
                output_path=tar_output,
                name=name,
                tag=tag,
                project_subpath=project_subpath,
                platform=platform,
            )
            print(f"\nbundle ready → {artifact}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run `agentix build` as a standalone entry point.

    Used by tests and by `python -m agentix.cli.build`; the console
    script routes through `agentix.cli:main` instead so subcommands
    share one help layout. Click usage errors are translated to
    `SystemExit`, matching the rest of the CLI.
    """
    try:
        build.main(args=argv, prog_name="agentix build", standalone_mode=False)
    except click.exceptions.UsageError as exc:
        exc.show(file=sys.stderr)
        raise SystemExit(exc.exit_code) from exc
    return 0


__all__ = ["build", "main"]
