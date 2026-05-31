"""Tests for `agentix plugin list`."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

import agentix.cli.plugin as plugin_mod


class _FakeRegistry:
    def __init__(self, loaded: dict | None = None, errors: dict | None = None) -> None:
        self._loaded = loaded or {}
        self._errors = errors or {}

    def all(self) -> dict:
        return dict(self._loaded)

    def errors(self) -> dict:
        return dict(self._errors)

    def sources(self) -> dict:
        return {}


@dataclass
class _FakeDist:
    name: str
    version: str


@dataclass
class _FakeEntryPoint:
    name: str
    value: str
    dist: _FakeDist | None


def _set_nix_entry_points(monkeypatch: pytest.MonkeyPatch, eps: Iterable[_FakeEntryPoint]) -> None:
    """Patch `importlib.metadata.entry_points` to return `eps` for the nix group only.

    The plugin command always queries the nix group by keyword (`group=`), so a
    minimal stub that filters on `group` is enough to drive the listing without
    touching the real install's entry points.
    """
    eps_list = list(eps)

    def _patched(*, group: str) -> Iterable[_FakeEntryPoint]:
        return eps_list if group == plugin_mod.NIX_GROUP else ()

    monkeypatch.setattr(plugin_mod.md, "entry_points", _patched)


def test_plugin_list_reports_loaded_and_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = _FakeRegistry(
        loaded={"docker": object, "podman": object},
        errors={"broken": RuntimeError("bad import")},
    )
    monkeypatch.setattr(plugin_mod, "providers", lambda: registry)
    _set_nix_entry_points(monkeypatch, [])

    assert plugin_mod.main(["list"]) == 0

    out = capsys.readouterr().out
    assert "providers (host-side sandbox backends)" in out
    assert "docker" in out and "ok" in out
    assert "podman" in out
    assert "broken" in out and "ERROR" in out and "RuntimeError" in out


def test_plugin_list_reports_nix_closures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(plugin_mod, "providers", lambda: _FakeRegistry())
    _set_nix_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("bash", "agentix.bash", _FakeDist("agentix-runtime-basic", "0.2.7")),
            _FakeEntryPoint("files", "agentix.files", _FakeDist("agentix-runtime-basic", "0.2.7")),
            _FakeEntryPoint(
                "claude-code",
                "agentix.agents.claude_code",
                _FakeDist("agentix-agent-claude-code", "0.2.7"),
            ),
        ],
    )

    assert plugin_mod.main(["list"]) == 0

    out = capsys.readouterr().out
    assert "nix closures (sandbox-side system deps)" in out
    assert "bash" in out and "agentix.bash" in out
    assert "files" in out and "agentix.files" in out
    assert "claude-code" in out and "agentix-agent-claude-code@0.2.7" in out
    # No providers installed — the provider section reports its own empty state,
    # not the global empty-state hint reserved for "nothing in any axis".
    assert "none installed" in out
    assert "no Agentix plugins installed" not in out


def test_plugin_list_lists_both_axes_together(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        plugin_mod,
        "providers",
        lambda: _FakeRegistry(loaded={"docker": object}),
    )
    _set_nix_entry_points(
        monkeypatch,
        [_FakeEntryPoint("bash", "agentix.bash", _FakeDist("agentix-runtime-basic", "0.2.7"))],
    )

    assert plugin_mod.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "providers (host-side sandbox backends)" in out
    assert "nix closures (sandbox-side system deps)" in out
    assert "none installed" not in out


def test_plugin_list_empty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(plugin_mod, "providers", lambda: _FakeRegistry())
    _set_nix_entry_points(monkeypatch, [])

    assert plugin_mod.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "no Agentix plugins installed" in out
    assert "providers" in out and "nix closures" in out
    assert "agentix-provider-docker" in out
    assert "agentix-runtime-basic" in out
