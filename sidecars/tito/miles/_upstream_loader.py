"""Helpers for delegating compatibility wrappers to an installed Miles tree."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class UpstreamModuleLoadError(ImportError):
    """Raised when a present upstream module cannot be imported."""


def _candidate_files(module_name: str, search_root: Path) -> tuple[Path, Path]:
    module_parts = module_name.split(".")
    module_path = search_root.joinpath(*module_parts)
    return module_path.with_suffix(".py"), module_path / "__init__.py"


def load_upstream_module(module_name: str, local_file: str) -> ModuleType | None:
    """Load an upstream Miles module with the same public name, if available.

    The local compatibility package intentionally occupies `miles.*` import
    paths.  Exact-name wrappers use this function to look past the current repo
    and delegate to a real upstream Miles installation when one is present.
    """
    local_path = Path(local_file).resolve()
    for entry in sys.path:
        search_root = Path(entry or ".").resolve()
        for candidate in _candidate_files(module_name, search_root):
            try:
                candidate = candidate.resolve()
            except OSError:
                continue
            if not candidate.exists() or candidate == local_path:
                continue

            digest = hashlib.sha1(str(candidate).encode("utf-8")).hexdigest()[:12]
            alias = f"_tito_gateway_upstream_{module_name.replace('.', '_')}_{digest}"
            if alias in sys.modules:
                return sys.modules[alias]

            is_package = candidate.name == "__init__.py"
            spec = importlib.util.spec_from_file_location(
                alias,
                candidate,
                submodule_search_locations=[str(candidate.parent)] if is_package else None,
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                sys.modules.pop(alias, None)
                raise UpstreamModuleLoadError(
                    f"Found upstream candidate for {module_name} at {candidate}, "
                    "but importing it failed. Fix the upstream Miles installation "
                    "or remove it from sys.path."
                ) from exc
            return module
    return None


def export_public(module: ModuleType, namespace: dict[str, object]) -> list[str]:
    """Copy public symbols from `module` into `namespace`."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [name for name in vars(module) if not name.startswith("_")]
    for name in names:
        namespace[name] = getattr(module, name)
    return list(names)
