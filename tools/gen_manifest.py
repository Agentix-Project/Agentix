"""Generate a `manifest.json` for a closure from its `__init__.py` metadata.

Run at build time, e.g. from a closure's `default.nix` postInstall:

    python tools/gen_manifest.py \\
        --init <path to closure __init__.py> \\
        --out  <path to manifest.json>

The script extracts metadata via `ast` without importing the closure
package, so it runs without the closure's runtime deps installed:

  * `name` and `package` are derived from the file path — anywhere the
    file sits under an `agentix_closures/<name>/` directory yields the
    expected package import path.
  * `version` comes from `__version__ = "..."` in the source.
  * `description` is the first non-empty line of the module docstring.
  * `abi` is the framework constant baked in here so nix doesn't need
    the `agentix` package available at build time. Bump in sync with
    `agentix.models.AGENTIX_CLOSURE_ABI` on protocol-breaking changes.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# Must match `agentix.models.AGENTIX_CLOSURE_ABI`. Duplicated here so the
# build step doesn't require the framework on its PYTHONPATH; promote
# this to an import once gen_manifest lives inside the framework wheel.
AGENTIX_CLOSURE_ABI = 1


def _extract_string_assign(tree: ast.Module, name: str) -> str | None:
    """Find a top-level `<name> = "..."` assignment and return its value."""
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def _derive_package(init_py: Path) -> str:
    """Map `<...>/agentix_closures/<name>/__init__.py` to `agentix_closures.<name>`.

    Accepts any depth above `agentix_closures` (development tree, nix
    site-packages, docker image layout — all work uniformly).
    """
    parts = init_py.resolve().parts
    if "agentix_closures" not in parts:
        raise SystemExit(
            f"{init_py}: not under an `agentix_closures/<name>/` directory"
        )
    i = parts.index("agentix_closures")
    return ".".join(parts[i:-1])  # strip trailing __init__.py


def generate(init_py: Path, package: str | None = None) -> dict[str, object]:
    """Return the manifest dict the runtime expects at `entry/manifest.json`."""
    src = init_py.read_text()
    tree = ast.parse(src)

    pkg = package or _derive_package(init_py)
    version = _extract_string_assign(tree, "__version__")
    if version is None:
        raise SystemExit(f"{init_py}: missing top-level `__version__ = '...'`")

    doc = ast.get_docstring(tree) or ""
    first_line = next((line.strip() for line in doc.splitlines() if line.strip()), "")

    manifest: dict[str, object] = {
        "abi": AGENTIX_CLOSURE_ABI,
        "name": pkg.rsplit(".", 1)[-1],
        "version": version,
        "package": pkg,
    }
    if first_line:
        manifest["description"] = first_line
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", type=Path, required=True,
                        help="path to the closure's __init__.py")
    parser.add_argument("--out", type=Path, required=True,
                        help="manifest.json output path")
    parser.add_argument("--package", type=str, default=None,
                        help="override the package import path (else inferred)")
    args = parser.parse_args(argv)

    manifest = generate(args.init, package=args.package)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out} for {manifest['package']} v{manifest['version']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
