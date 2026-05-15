"""Spec resolution — chain-of-responsibility plugin axis.

A *spec* is whatever the user types on the command line: a short name
(`bash`), a relative path (`./primitives/bash`), or an image reference
(`docker.io/me/agent:0.1.0`). The framework asks each registered
**spec resolver** in priority order to map the string to a
`ClosureSpec`; the first non-`None` answer wins.

Resolvers register under the `agentix.spec_resolver` entry-point group.
Each entry value is a `module:Resolver` class implementing the
`SpecResolver` Protocol. Builtin resolvers live in this module
(path, image-ref, local-roots fallback, PyPI fallback) and ship via
the framework's own `pyproject.toml`.

```toml
# downstream pyproject.toml
[project.entry-points."agentix.spec_resolver"]
github = "my_agentix_github_resolver:GithubResolver"
```

```python
# downstream module
class GithubResolver:
    priority = 30  # higher → tried earlier

    def resolve(self, spec: str) -> ClosureSpec | None:
        if not spec.startswith("github:"):
            return None
        ...
```

After `pip install`, `agentix build github:my-org/closure` finds the
resolver and uses it without framework changes.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from agentix._plugin import Registry

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ClosureSpec:
    """One resolved input to a build / install. Exactly one source field is set."""

    short: str
    kind: Literal["path", "pypi", "image"]
    path: Path | None = None
    pypi_dist: str | None = None
    image_ref: str | None = None


@runtime_checkable
class SpecResolver(Protocol):
    """Plugin contract for closure-spec resolvers.

    Higher `priority` values are tried first. Two resolvers with the same
    priority fall back to entry-point name order (asc).
    """

    priority: int

    def resolve(self, spec: str) -> ClosureSpec | None:
        """Return a `ClosureSpec` if this resolver claims the spec; else None."""
        ...


# Plugin registry — `agentix.spec_resolver` group. Built-in resolvers
# are registered in the framework's pyproject.toml.
_resolvers: Registry[type[SpecResolver]] = Registry("agentix.spec_resolver")


def register_spec_resolver(name: str, cls: type[SpecResolver]) -> None:
    """In-process resolver registration. Tests / dynamic use only."""
    _resolvers.register(name, lambda: cls)


def spec_resolvers() -> Registry[type[SpecResolver]]:
    """The underlying registry — for `agentix plugins` and tests."""
    return _resolvers


def _chain() -> list[SpecResolver]:
    """Snapshot of every registered resolver, ordered by priority desc.

    Loaded lazily on every call so `register_spec_resolver` mid-process
    takes effect immediately (matters for test fixtures).
    """
    instances: list[tuple[int, str, SpecResolver]] = []
    for name, cls in _resolvers.all().items():
        inst = cls()
        priority = getattr(inst, "priority", 0)
        instances.append((-priority, name, inst))
    instances.sort(key=lambda t: (t[0], t[1]))  # desc priority, then name asc
    return [inst for _p, _n, inst in instances]


def resolve_spec(spec: str) -> ClosureSpec:
    """Walk every registered resolver in priority order; first match wins."""
    for resolver in _chain():
        result = resolver.resolve(spec)
        if result is not None:
            return result
    raise SystemExit(f"no spec resolver claimed {spec!r}")


def read_pyproject(closure_dir: Path) -> dict:
    pp = closure_dir / "pyproject.toml"
    if not pp.is_file():
        raise SystemExit(f"{closure_dir}: missing pyproject.toml")
    with pp.open("rb") as f:
        return tomllib.load(f)


# ── Built-in resolvers ──────────────────────────────────────────────


def _short_from_pyproject(pyproject: dict) -> str:
    """`agentix-bash` → `bash`."""
    name = pyproject.get("project", {}).get("name", "")
    if not isinstance(name, str) or not name.startswith("agentix-"):
        raise SystemExit(
            f"pyproject.toml: name {name!r} must start with `agentix-`"
        )
    return name[len("agentix-"):]


def _short_from_image(ref: str) -> str:
    """`docker.io/me/agentix/bash:0.1.0` → `bash`."""
    last = ref.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    return last[len("agentix-"):] if last.startswith("agentix-") else last


class PathResolver:
    """Treat explicit-path strings and existing source dirs as closure sources."""

    priority = 100

    def resolve(self, spec: str) -> ClosureSpec | None:
        if spec.startswith((".", "/")):
            p = Path(spec).resolve()
            if not (p / "pyproject.toml").is_file():
                raise SystemExit(
                    f"{spec}: no pyproject.toml — not a closure source dir"
                )
            return ClosureSpec(
                short=_short_from_pyproject(read_pyproject(p)),
                kind="path", path=p,
            )
        p = Path(spec)
        if p.is_dir() and (p / "pyproject.toml").is_file():
            return ClosureSpec(
                short=_short_from_pyproject(read_pyproject(p.resolve())),
                kind="path", path=p.resolve(),
            )
        return None


class ImageRefResolver:
    """`host/path:tag` strings — pre-built image references."""

    priority = 90

    def resolve(self, spec: str) -> ClosureSpec | None:
        if "/" in spec and ":" in spec and not spec.startswith((".", "/")):
            return ClosureSpec(
                short=_short_from_image(spec),
                kind="image", image_ref=spec,
            )
        return None


class LocalRepoResolver:
    """Short names looked up under the repo's `primitives/<name>/` tree.

    Kind-specific roots (`agents/`, `datasets/`, …) are added by
    downstream resolvers; the framework only knows about `primitives/`.
    """

    priority = 50
    _roots: tuple[str, ...] = ("primitives",)

    def resolve(self, spec: str) -> ClosureSpec | None:
        for root in self._roots:
            candidate = REPO_ROOT / root / spec
            if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
                return ClosureSpec(
                    short=spec, kind="path", path=candidate,
                )
        return None


class PyPIFallbackResolver:
    """Last-chance: assume the bare name is a published PyPI dist."""

    priority = 10

    def resolve(self, spec: str) -> ClosureSpec | None:
        # If we got this far the spec wasn't a path, image ref, or local
        # closure. Treat it as a PyPI distribution name. The actual fetch
        # is stubbed (build/install raise NotImplementedError at stage time).
        return ClosureSpec(
            short=spec, kind="pypi", pypi_dist=f"agentix-{spec}",
        )


__all__ = [
    "REPO_ROOT",
    "ClosureSpec",
    "ImageRefResolver",
    "LocalRepoResolver",
    "PathResolver",
    "PyPIFallbackResolver",
    "SpecResolver",
    "read_pyproject",
    "register_spec_resolver",
    "resolve_spec",
    "spec_resolvers",
]
