"""Plugin registry for the `agentix.provider` entry-point group.

Lives under `agentix.provider` because deployments are the only
framework axis discovered via entry points. Remote callables do not need
a framework registry; only sandbox lifecycle backends need a name-to-class
registry.

Two ways to find plugins:

  * **Production:** `importlib.metadata.entry_points(group=...)` — what
    `pip install some-plugin` populates. The dist's `pyproject.toml`
    declares its entries; nothing in the framework changes when a new
    plugin is installed.
  * **Testing / dynamic:** the in-process `register(name, factory)`
    method. Convenient for unit tests, fixtures, or programmatic
    composition; not part of the documented user-facing surface.

Lookup is lazy — entry points are walked on the first `get()` /
`all()` call. Loaders that raise are caught and remembered per-entry
(`errors()`), so one broken plugin doesn't poison the rest.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

logger = logging.getLogger("agentix.provider.plugin")

T = TypeVar("T")


@dataclass(frozen=True)
class PluginSource:
    """Where a registered plugin came from — used in conflict reports
    and CLI listings."""

    dist_name: str | None
    dist_version: str | None

    def label(self) -> str:
        if self.dist_name:
            return f"{self.dist_name}@{self.dist_version or '?'}"
        return "(in-process)"


class PluginConflictError(RuntimeError):
    """Two plugins claim the same name in the same group.

    The framework refuses to silently last-wins because it would hide
    bugs (e.g. a stale wheel from a previous install). Users see the
    conflict on first registry access and uninstall the duplicate dist.
    """


class Registry(Generic[T]):
    """One `agentix.<axis>` entry-point group + in-process `register()`.

    `T` is whatever the entry-point load resolves to — typically a class
    (`type[SandboxProvider]`) or a callable factory. The registry doesn't
    instantiate anything; callers decide what to do with the loaded
    value. This keeps the contract narrow: T's shape is the axis's
    Protocol; how it gets instantiated is the axis's concern.
    """

    def __init__(self, group: str) -> None:
        self._group = group
        self._extra: dict[str, tuple[Callable[[], T], PluginSource]] = {}
        # Lazy: populated on first _load().
        self._cache: dict[str, T] | None = None
        self._sources: dict[str, PluginSource] = {}
        self._errors: dict[str, Exception] = {}

    @property
    def group(self) -> str:
        return self._group

    def register(
        self,
        name: str,
        factory: Callable[[], T],
        *,
        dist_name: str | None = None,
        dist_version: str | None = None,
    ) -> None:
        """Register a plugin imperatively.

        Intended for tests and programmatic composition; production
        plugins use entry points. Calling `register()` invalidates the
        cache so the next lookup re-runs the merge.
        """
        self._extra[name] = (factory, PluginSource(dist_name, dist_version))
        self._cache = None  # invalidate

    def _walk_entry_points(self) -> list[tuple[str, Callable[[], T], PluginSource]]:
        eps = importlib.metadata.entry_points()
        # Python 3.10+: SelectableGroups; earlier: dict (we target 3.11+ but
        # mirror the standard library's defensive branch anyway).
        selected = (
            list(eps.select(group=self._group)) if hasattr(eps, "select") else list(eps.get(self._group, []))  # type: ignore[attr-defined]
        )
        out: list[tuple[str, Callable[[], T], PluginSource]] = []
        for ep in selected:
            dist = ep.dist
            src = PluginSource(
                dist_name=getattr(dist, "name", None) if dist else None,
                dist_version=getattr(dist, "version", None) if dist else None,
            )
            out.append((ep.name, ep.load, src))
        return out

    def _load(self) -> dict[str, T]:
        if self._cache is not None:
            return self._cache

        items: dict[str, T] = {}
        sources: dict[str, PluginSource] = {}
        errors: dict[str, Exception] = {}

        # Entry-point pass first. Two dists declaring the same name is
        # ambiguous and must surface — they'd silently last-wins otherwise.
        for name, loader, src in self._walk_entry_points():
            if name in sources:
                raise PluginConflictError(
                    f"duplicate plugin {name!r} in group {self._group!r}: {sources[name].label()} vs {src.label()}"
                )
            try:
                items[name] = loader()
                sources[name] = src
            except Exception as exc:
                logger.warning(
                    "plugin %r in group %r failed to load: %s",
                    name,
                    self._group,
                    exc,
                )
                errors[name] = exc

        # In-process extras override entry points — this is deliberate
        # for tests (`register("local", FakeDocker)` swaps in a stub).
        # Production code paths don't call `register()`, so this is safe.
        for name, (factory, src) in self._extra.items():
            try:
                items[name] = factory()
                sources[name] = src
                errors.pop(name, None)
            except Exception as exc:
                errors[name] = exc

        self._cache = items
        self._sources = sources
        self._errors = errors
        return items

    def get(self, name: str) -> T:
        """Return the plugin registered under `name`.

        Raises `KeyError` if no plugin claims the name (with the list of
        available names in the error message), or re-raises the original
        exception if the named plugin failed to load.
        """
        items = self._load()
        if name in items:
            return items[name]
        if name in self._errors:
            raise self._errors[name]
        raise KeyError(f"no plugin {name!r} in group {self._group!r}; available: {sorted(items)}")

    def all(self) -> dict[str, T]:
        """Snapshot of all successfully-loaded plugins, name → value."""
        return dict(self._load())

    def sources(self) -> dict[str, PluginSource]:
        """`name → PluginSource` for every successfully-loaded plugin."""
        self._load()
        return dict(self._sources)

    def errors(self) -> dict[str, Exception]:
        """`name → Exception` for plugins whose load failed (cached)."""
        self._load()
        return dict(self._errors)

    def reset(self) -> None:
        """Test-only: drop cache + in-process registrations.

        Useful in pytest fixtures that need a known-empty registry
        between tests.
        """
        self._cache = None
        self._extra.clear()
        self._sources.clear()
        self._errors.clear()


__all__ = ["PluginConflictError", "PluginSource", "Registry"]
