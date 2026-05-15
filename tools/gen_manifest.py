"""Generate a `manifest.json` for a closure from its `pyproject.toml`.

Run at build time, e.g. from a closure's `default.nix` postInstall:

    python tools/gen_manifest.py \\
        --pyproject <path to pyproject.toml> \\
        --out       <path to manifest.json>

`pyproject.toml` is the single source of metadata truth: name, version,
description all come from `[project]`. The closure's Python import path
is derived from the wheel-packages list (`[tool.hatch.build.targets.wheel]
packages`), which is always exactly one `agentix_closures/<name>`.

The script uses stdlib only (`tomllib`, `json`) so the build step
doesn't need the framework's runtime deps installed.
"""

from __future__ import annotations

import argparse
import json
import sys

# `tomllib` is stdlib in 3.11+. Build environments at or above the
# framework's `requires-python = ">=3.11"` floor always have it.
import tomllib  # type: ignore[import-not-found]
from pathlib import Path

# Must match `agentix.models.AGENTIX_CLOSURE_ABI`. Duplicated here so the
# build step doesn't require the framework on its PYTHONPATH.
AGENTIX_CLOSURE_ABI = 1


def _derive_package(pyproject: dict) -> str:
    """Pull the import path from `[tool.hatch.build.targets.wheel] packages`.

    Convention: exactly one package, formatted `agentix_closures/<name>`.
    Returned as the dotted import form, e.g. `agentix_closures.bash`.
    """
    try:
        packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    except KeyError as exc:
        raise SystemExit(
            "pyproject.toml: [tool.hatch.build.targets.wheel] packages not set"
        ) from exc
    if not isinstance(packages, list) or len(packages) != 1:
        raise SystemExit(
            f"pyproject.toml: expected exactly one wheel package, got {packages!r}"
        )
    pkg_path = packages[0].replace("\\", "/").strip("/").replace("/", ".")
    if not pkg_path.startswith("agentix_closures."):
        raise SystemExit(
            f"pyproject.toml: wheel package {pkg_path!r} must live under "
            f"agentix_closures/<name>/"
        )
    return pkg_path


def generate(pyproject_path: Path) -> dict[str, object]:
    """Return the manifest dict the runtime expects at `entry/manifest.json`."""
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)
    project = pyproject.get("project", {})
    version = project.get("version")
    if not isinstance(version, str):
        raise SystemExit(f"{pyproject_path}: missing [project] version")
    description = project.get("description")
    pkg = _derive_package(pyproject)
    manifest: dict[str, object] = {
        "abi": AGENTIX_CLOSURE_ABI,
        "name": pkg.rsplit(".", 1)[-1],
        "version": version,
        "package": pkg,
    }
    if isinstance(description, str) and description.strip():
        manifest["description"] = description.strip()
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, required=True,
                        help="path to the closure's pyproject.toml")
    parser.add_argument("--out", type=Path, required=True,
                        help="manifest.json output path")
    args = parser.parse_args(argv)

    manifest = generate(args.pyproject)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out} for {manifest['package']} v{manifest['version']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
