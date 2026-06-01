"""In-container build step — collect every system-deps Nix closure.

This runs *inside* the `agentix build` container, after `uv sync` has
materialized the project's full dependency closure into the venv. At
that point every plugin is installed and introspectable, so this step:

  1. Discovers plugin Nix closures. A plugin declares one entry point
     in the `agentix.nix` group; its value names the module that ships
     a `default.nix` as package data. Walking the group enumerates
     only packages that *registered* — discovery follows the
     dependency graph, with provenance (which distribution shipped
     each file). No directory guessing.

  2. Reads the project's own closure from `[tool.agentix] nix` — the
     one place a bundle author writes Nix. Optional.

  3. Stages each plugin's `.nix` into `closures/plugins/<label>.nix`,
     where the flake's `plugins` output imports each as its own
     derivation (no merge across plugins). The project's `.nix`
     (if declared) lands at `closures/project.nix` and is merged
     into the flake's `runtime` output alongside the toolchain.

Plugin discovery walks `importlib.metadata.entry_points` (reading
`.dist-info/entry_points.txt`) and locates each closure's `default.nix`
as package data. An editable / workspace install is resolved from its
`direct_url.json` without importing; for a wheel / registry install,
`importlib.resources` imports the entry-point module to find its data
dir. A plugin whose `direct_url.json` is malformed or whose module fails
to import is skipped (logged), never fatal — discovery of the rest
continues.

Invoked from `bundle-build.sh` as:

    python -m agentix.cli.build.closures --project P --closures C

There's no user-facing command for this — it's an internal step that
just happens to be a Python module so it can use the same plugin-
discovery code path the host build relies on.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from urllib.parse import unquote, urlparse

import click

from agentix.cli.build.pyproject import project_nix, read_pyproject

logger = logging.getLogger("agentix.cli.build.closures")

# Entry-point group a plugin registers to declare it ships a Nix
# closure. The entry-point *value* is the module under which a
# `default.nix` rides as package data.
NIX_ENTRY_POINT_GROUP = "agentix.nix"

# Filename a contributing module ships next to its package data.
CLOSURE_FILENAME = "default.nix"


@dataclass(frozen=True)
class Closure:
    """One discovered `{ pkgs }: drv` Nix file ready to stage."""

    label: str  # unique, filesystem-safe — used as the staged filename stem
    origin: str  # human-readable provenance, for logs
    content: bytes


def discover_plugin_closures() -> list[Closure]:
    """Every plugin-contributed Nix closure in the current environment.

    For each `agentix.nix` entry point, the value names a module; if
    that module ships a `default.nix` as package data, it is collected.
    A plugin that registers the entry point but ships no file is
    skipped (not an error — `importlib.resources` simply finds nothing).
    """
    found: list[Closure] = []
    seen: set[str] = set()
    for ep in sorted(md.entry_points(group=NIX_ENTRY_POINT_GROUP), key=lambda e: e.name):
        dist = getattr(ep, "dist", None)
        dist_name = getattr(dist, "name", None) or "unknown"
        label = f"{dist_name}.{ep.name}".replace("/", "_")
        if label in seen:
            continue
        nix_file = _plugin_nix_file(ep)
        if nix_file is None:
            continue
        seen.add(label)
        found.append(
            Closure(
                label=label,
                origin=f"plugin {dist_name} ({nix_file})",
                content=nix_file.read_bytes(),
            )
        )
    return found


def _plugin_nix_file(ep: md.EntryPoint):
    dist = getattr(ep, "dist", None)
    direct_url = dist.read_text("direct_url.json") if dist is not None else None
    if direct_url:
        # `direct_url.json` is occasionally malformed (older installers); a
        # parse failure must not abort discovery — fall through to package data.
        try:
            url = json.loads(direct_url).get("url", "")
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("ignoring malformed direct_url.json for %r", ep.value)
            url = ""
        parsed = urlparse(url)
        if parsed.scheme == "file":
            project_dir = Path(unquote(parsed.path)).resolve()
            if (project_dir / "pyproject.toml").is_file():
                rel = project_nix(read_pyproject(project_dir))
                if rel is not None:
                    return (project_dir / rel).resolve()

    # Locating package data imports the entry-point module (this is also what
    # resolves editable installs). A plugin with a heavy or broken top-level
    # import must be skipped, not abort discovery of every other closure.
    try:
        nix_file = resources.files(ep.value) / CLOSURE_FILENAME
        return nix_file if nix_file.is_file() else None
    except Exception:
        logger.warning("skipping plugin %r: could not locate its package data", ep.value, exc_info=True)
        return None


def discover_project_closure(project_dir: Path) -> Closure | None:
    """The project's own closure, if it declares `[tool.agentix] nix`.

    The path is resolved relative to the project root and must stay
    inside it — a bundle's Nix file is part of the project, not a
    pointer into the wider filesystem.
    """
    pyproject = read_pyproject(project_dir)
    rel = project_nix(pyproject)
    if rel is None:
        return None
    project_dir = project_dir.resolve()
    nix_file = (project_dir / rel).resolve()
    if not str(nix_file).startswith(str(project_dir) + "/"):
        raise SystemExit(f"[tool.agentix].nix {rel!r} escapes the project directory")
    if not nix_file.is_file():
        raise SystemExit(f"[tool.agentix].nix points at {rel!r} — no such file")
    return Closure(
        label="project",
        origin=f"project ({rel})",
        content=nix_file.read_bytes(),
    )


def stage_closures(closures: Sequence[Closure], closures_dir: Path) -> list[Path]:
    """Write every closure into `closures_dir` as `<label>.nix`."""
    closures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for closure in closures:
        dest = closures_dir / f"{closure.label}.nix"
        dest.write_bytes(closure.content)
        written.append(dest)
    return written


def assemble(project_dir: Path, closures_dir: Path) -> list[Closure]:
    """Discover plugin + project closures and stage them.

    Plugins go to `closures_dir/plugins/<label>.nix` — the flake reads
    each as its own derivation (no merge across plugins). The project
    closure goes to `closures_dir/project.nix` — merged into the
    runtime tree. Returns every staged closure for the caller to log.
    """
    plugins = discover_plugin_closures()
    stage_closures(plugins, closures_dir / "plugins")

    project = discover_project_closure(project_dir)
    if project is not None:
        stage_closures([project], closures_dir)

    out = list(plugins)
    if project is not None:
        out.append(project)
    return out


@click.command(
    name="closures",
    help="Collect system-deps Nix closures (internal build step).",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--project",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Project root.",
)
@click.option(
    "--closures",
    "closures_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output dir for staged .nix files.",
)
def main(project: Path, closures_dir: Path) -> None:
    """Discover plugin + project closures and stage them into `closures_dir`."""
    collected = assemble(project.resolve(), closures_dir.resolve())
    if not collected:
        print("no system-deps closures — bundle is pure-Python")
    for closure in collected:
        print(f"  closure: {closure.label}.nix  ←  {closure.origin}")


if __name__ == "__main__":
    main()
