"""Tests for `agentix.deployment._plugin.Registry[T]`.

Covers the four scenarios every plugin axis cares about:

  * happy-path lookup via entry points
  * happy-path lookup via in-process `register()`
  * duplicate-name conflict (entry points)
  * one plugin failing to load doesn't poison the rest
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agentix.deployment._plugin import PluginConflictError, Registry


@dataclass(frozen=True)
class _FakeDist:
    name: str
    version: str | None


@dataclass(frozen=True)
class _FakeEntryPoint:
    name: str
    load: object
    value: str
    dist: _FakeDist | None


def _fake_ep(name: str, loader, dist_name: str | None = None, dist_version: str | None = None):
    """Build a stand-in for an importlib.metadata EntryPoint."""
    dist = _FakeDist(name=dist_name, version=dist_version) if dist_name is not None else None
    return _FakeEntryPoint(name=name, load=loader, value=f"x:{name}", dist=dist)


def _patch_eps(registry: Registry, eps: list):
    """Replace the registry's entry-point walker with a synthetic list.

    Avoids the cost (and global-state risk) of actually installing fake
    distributions into the test env's site-packages.
    """
    items = [
        (ep.name, ep.load, _src(ep.dist.name if ep.dist else None, ep.dist.version if ep.dist else None)) for ep in eps
    ]
    registry._walk_entry_points = lambda: items  # type: ignore[method-assign]


def _src(name, version):
    from agentix.deployment._plugin import PluginSource

    return PluginSource(dist_name=name, dist_version=version)


def test_entry_point_happy_path():
    reg = Registry("test.axis")
    _patch_eps(reg, [_fake_ep("a", lambda: "value-a", "dist-a", "1.0")])
    assert reg.get("a") == "value-a"
    assert reg.all() == {"a": "value-a"}


def test_in_process_register():
    reg = Registry("test.axis")
    _patch_eps(reg, [])
    reg.register("b", lambda: "value-b", dist_name="(test)", dist_version="0.0")
    assert reg.get("b") == "value-b"


def test_in_process_register_overrides_entry_point():
    """register() lets tests swap in a stub for a plugin that's also
    declared as an entry point. Useful for monkey-patching a real
    backend with a fake during tests."""
    reg = Registry("test.axis")
    _patch_eps(reg, [_fake_ep("local", lambda: "real-impl", "real-dist", "1.0")])
    reg.register("local", lambda: "fake-impl")
    assert reg.get("local") == "fake-impl"


def test_entry_point_duplicate_raises_conflict():
    """Two dists registering the same name in the same group must
    surface as an error — silent last-wins would hide a stale install."""
    reg = Registry("test.axis")
    _patch_eps(
        reg,
        [
            _fake_ep("local", lambda: "a", "dist-a", "1.0"),
            _fake_ep("local", lambda: "b", "dist-b", "2.0"),
        ],
    )
    with pytest.raises(PluginConflictError) as excinfo:
        reg.all()
    msg = str(excinfo.value)
    assert "dist-a@1.0" in msg
    assert "dist-b@2.0" in msg


def test_loader_failure_does_not_poison_others():
    """A plugin whose loader raises is cached as an error; sibling plugins
    still load."""

    def bad_loader():
        raise RuntimeError("kaboom")

    reg = Registry("test.axis")
    _patch_eps(
        reg,
        [
            _fake_ep("good", lambda: "ok"),
            _fake_ep("bad", bad_loader, "broken-dist", "0.0.1"),
        ],
    )
    assert reg.get("good") == "ok"
    assert reg.all() == {"good": "ok"}

    errors = reg.errors()
    assert "bad" in errors
    assert "kaboom" in str(errors["bad"])

    # Asking for the broken plugin re-raises the cached error.
    with pytest.raises(RuntimeError, match="kaboom"):
        reg.get("bad")


def test_get_unknown_name_lists_available():
    reg = Registry("test.axis")
    _patch_eps(reg, [_fake_ep("a", lambda: 1), _fake_ep("b", lambda: 2)])
    with pytest.raises(KeyError) as excinfo:
        reg.get("c")
    msg = str(excinfo.value)
    assert "'a'" in msg and "'b'" in msg


def test_register_invalidates_cache():
    reg = Registry("test.axis")
    _patch_eps(reg, [_fake_ep("a", lambda: 1)])
    assert reg.all() == {"a": 1}
    reg.register("b", lambda: 2)
    assert reg.all() == {"a": 1, "b": 2}


def test_sources_reports_origin():
    reg = Registry("test.axis")
    _patch_eps(reg, [_fake_ep("ep", lambda: "x", "ep-dist", "1.2")])
    reg.register("manual", lambda: "y")
    srcs = reg.sources()
    assert srcs["ep"].label() == "ep-dist@1.2"
    assert srcs["manual"].label() == "(in-process)"


def test_reset_clears_cache_and_extras():
    reg = Registry("test.axis")
    _patch_eps(reg, [_fake_ep("a", lambda: 1)])
    reg.register("b", lambda: 2)
    assert reg.all() == {"a": 1, "b": 2}
    reg.reset()
    # After reset, only entry-point side remains (extras dropped).
    assert reg.all() == {"a": 1}
