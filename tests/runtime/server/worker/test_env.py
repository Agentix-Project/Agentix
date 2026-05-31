from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentix.runtime.server.worker.client import _clean_worker_env
from agentix.runtime.shared.env import AGENTIX_ADDED_LD_LIBRARY_PATH, AGENTIX_ADDED_PATH, BUNDLE_RUNTIME_BIN


def test_clean_worker_env_inherits_parent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["/custom/bin", "/usr/bin"]))

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert env["PATH"].split(os.pathsep) == [
        "/runtime/venv/bin",
        BUNDLE_RUNTIME_BIN,
        "/custom/bin",
        "/usr/bin",
    ]
    assert env[AGENTIX_ADDED_PATH].split(os.pathsep) == [
        "/runtime/venv/bin",
        BUNDLE_RUNTIME_BIN,
    ]


def test_clean_worker_env_dedupes_prepended_path_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([BUNDLE_RUNTIME_BIN, "/usr/bin", "/runtime/venv/bin"]),
    )

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert env["PATH"].split(os.pathsep) == [
        "/runtime/venv/bin",
        BUNDLE_RUNTIME_BIN,
        "/usr/bin",
    ]
    assert env[AGENTIX_ADDED_PATH].split(os.pathsep) == [
        "/runtime/venv/bin",
        BUNDLE_RUNTIME_BIN,
    ]


def test_clean_worker_env_preserves_inherited_agentix_added_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["/custom/bin", "/usr/bin"]))
    monkeypatch.setenv(AGENTIX_ADDED_PATH, "/already/added")

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert env[AGENTIX_ADDED_PATH].split(os.pathsep) == [
        "/already/added",
        "/runtime/venv/bin",
        BUNDLE_RUNTIME_BIN,
    ]


def test_clean_worker_env_injects_recorded_runtime_build_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "PKG_CONFIG_PATH",
        "CMAKE_PREFIX_PATH",
        "AGENTIX_ADDED_LD_LIBRARY_PATH",
        "AGENTIX_ADDED_LIBRARY_PATH",
        "AGENTIX_ADDED_CPATH",
        "AGENTIX_ADDED_C_INCLUDE_PATH",
        "AGENTIX_ADDED_CPLUS_INCLUDE_PATH",
        "AGENTIX_ADDED_PKG_CONFIG_PATH",
        "AGENTIX_ADDED_CMAKE_PREFIX_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/task/lib")
    monkeypatch.setenv("PKG_CONFIG_PATH", "/task/lib/pkgconfig")

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == ["/nix/runtime/lib", "/task/lib"]
    assert env[AGENTIX_ADDED_LD_LIBRARY_PATH] == "/nix/runtime/lib"
    assert env["LIBRARY_PATH"] == "/nix/runtime/lib"
    assert env["CPATH"] == "/nix/runtime/include"
    assert env["C_INCLUDE_PATH"] == "/nix/runtime/include"
    assert env["CPLUS_INCLUDE_PATH"] == "/nix/runtime/include"
    assert env["PKG_CONFIG_PATH"].split(os.pathsep) == [
        "/nix/runtime/lib/pkgconfig",
        "/nix/runtime/share/pkgconfig",
        "/task/lib/pkgconfig",
    ]
    assert env["CMAKE_PREFIX_PATH"] == "/nix/runtime"
