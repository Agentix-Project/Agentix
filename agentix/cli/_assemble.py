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

  3. Stages every `.nix` file into `closures/`, where the flake's
     `runtime` output imports and `symlinkJoin`s them.

Plugin discovery is import-free: `importlib.metadata.entry_points`
reads `.dist-info/entry_points.txt`; `importlib.resources` locates the
file. Neither imports plugin code — safe to run over an arbitrary venv.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from urllib.parse import unquote, urlparse

from agentix.cli._resolve import project_nix, read_pyproject

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
    if direct_url is not None:
        parsed = urlparse(json.loads(direct_url).get("url", ""))
        if parsed.scheme == "file":
            project_dir = Path(unquote(parsed.path)).resolve()
            if (project_dir / "pyproject.toml").is_file():
                rel = project_nix(read_pyproject(project_dir))
                if rel is not None:
                    return (project_dir / rel).resolve()

    try:
        nix_file = resources.files(ep.value) / CLOSURE_FILENAME
    except (ModuleNotFoundError, TypeError):
        return None
    return nix_file if nix_file.is_file() else None


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
    """Discover plugin + project closures and stage them. Returns the
    closures staged, for the caller to log."""
    closures = discover_plugin_closures()
    project = discover_project_closure(project_dir)
    if project is not None:
        closures.append(project)
    stage_closures(closures, closures_dir)
    return closures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentix._assemble",
        description="collect system-deps Nix closures (internal build step)",
    )
    parser.add_argument("--project", required=True, type=Path, help="project root")
    parser.add_argument("--closures", required=True, type=Path, help="output dir for staged .nix files")
    args = parser.parse_args(argv)

    closures = assemble(args.project.resolve(), args.closures.resolve())
    if not closures:
        print("no system-deps closures — bundle is pure-Python")
    for closure in closures:
        print(f"  closure: {closure.label}.nix  ←  {closure.origin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
