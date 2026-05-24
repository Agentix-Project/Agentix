"""Read a project's `pyproject.toml` and derive build metadata.

`agentix build` takes one project root — a directory containing
`pyproject.toml` + `uv.lock`. Plugins (other `agentix-*` packages) are
pulled in transitively via the lock; neither the CLI nor the user
enumerates them on the command line.

This module owns the small bit of metadata extraction the build needs:

  * `read_pyproject(path)` — parse the project's pyproject.toml.
  * `short_name(pyproject)` — display/tag short name.
  * `derive_tag(pyproject)` — `<short>:<version>`.
  * `detect_python_version(pyproject)` — Nixpkgs python attr suffix.
  * `project_nix(pyproject)` — the user's own `[tool.agentix] nix` file
    path, if declared.

There's no multi-spec resolver, no PyPI fallback. The spec is always a
local project root.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Python minor versions Nixpkgs ships an attribute for. The build can
# only materialize an interpreter that exists as `pkgs.python3<minor>`.
_SUPPORTED_PY_MINORS = range(10, 14)
_DEFAULT_PY = "311"


def read_pyproject(project_dir: Path) -> dict:
    pp = project_dir / "pyproject.toml"
    if not pp.is_file():
        raise SystemExit(f"{project_dir}: missing pyproject.toml")
    with pp.open("rb") as f:
        return tomllib.load(f)


def short_name(pyproject: dict) -> str:
    """Derive a short display/tag name for the project.

    The short name only affects the bundle tag and a few build
    diagnostics — wire routing is by `fn.__module__`, which is
    determined by the user's actual Python import path.
    """
    project = pyproject.get("project", {})
    name = project.get("name", "")
    if not isinstance(name, str) or not name:
        raise SystemExit("pyproject.toml: [project].name is required")
    return name.removeprefix("agentix-")


def derive_tag(pyproject: dict) -> str:
    """`<short>:<version>` from the pyproject."""
    project = pyproject.get("project", {})
    version = project.get("version")
    if not isinstance(version, str):
        raise SystemExit("pyproject.toml: [project].version is required")
    return f"{short_name(pyproject)}:{version}"


def detect_python_version(pyproject: dict) -> str:
    """Return the Nixpkgs python attr suffix (e.g. `311`, `312`, `313`).

    `agentix build` runs on the host with no project venv, so there is
    no live interpreter to read — the version comes from the project's
    `[project].requires-python` lower bound. The lock was resolved
    against that bound, so the bundle's interpreter matches it.

    Defaults to `311` when the bound is missing or unparseable.
    """
    req = pyproject.get("project", {}).get("requires-python", "")
    if not isinstance(req, str):
        return _DEFAULT_PY
    for token in req.replace(",", " ").split():
        token = token.lstrip(">=~^! ")
        if not token.startswith("3."):
            continue
        try:
            minor = int(token.split(".")[1].rstrip(".*"))
        except (ValueError, IndexError):
            continue
        if minor in _SUPPORTED_PY_MINORS:
            return f"3{minor}"
    return _DEFAULT_PY


def project_nix(pyproject: dict) -> str | None:
    """The project's own system-deps Nix file, declared as
    `[tool.agentix] nix = "<path>"` — a relative path to a
    `{ pkgs }: drv` file. Returns None when not declared.

    This is the *only* place a bundle author touches Nix. Plugin Nix
    files are discovered automatically (see `agentix.cli.build.closures`)
    and never surface to the project author.
    """
    value = pyproject.get("tool", {}).get("agentix", {}).get("nix")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SystemExit("pyproject.toml: [tool.agentix].nix must be a non-empty path string")
    return value


__all__ = [
    "REPO_ROOT",
    "derive_tag",
    "detect_python_version",
    "project_nix",
    "read_pyproject",
    "short_name",
]
