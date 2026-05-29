"""Tests for `agentix plugin list`."""

from __future__ import annotations

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


def test_plugin_list_reports_loaded_and_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = _FakeRegistry(
        loaded={"docker": object, "podman": object},
        errors={"broken": RuntimeError("bad import")},
    )
    monkeypatch.setattr(plugin_mod, "providers", lambda: registry)

    assert plugin_mod.main(["list"]) == 0

    out = capsys.readouterr().out
    assert "docker" in out and "ok" in out
    assert "podman" in out
    assert "broken" in out and "ERROR" in out and "RuntimeError" in out


def test_plugin_list_empty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(plugin_mod, "providers", lambda: _FakeRegistry())
    assert plugin_mod.main(["list"]) == 0
    assert "no deployment backends installed" in capsys.readouterr().out
