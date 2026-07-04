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


def test_clean_worker_env_records_stripped_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stripping without recording makes the image env unrecoverable for task
    subprocesses; every stripped var must land in AGENTIX_SAVED_*."""
    monkeypatch.setenv("PYTHONPATH", "/testbed/src")
    monkeypatch.setenv("LD_PRELOAD", "/usr/lib/libfoo.so")
    monkeypatch.setenv("NIX_CFLAGS_COMPILE", "-O2")

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert "PYTHONPATH" not in env
    assert env["AGENTIX_SAVED_PYTHONPATH"] == "/testbed/src"
    assert env["AGENTIX_SAVED_LD_PRELOAD"] == "/usr/lib/libfoo.so"
    assert env["AGENTIX_SAVED_NIX_CFLAGS_COMPILE"] == "-O2"


def test_clean_worker_env_records_nothing_when_nothing_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    # The runner's own environment may legitimately carry strippable vars
    # (NIX_*, SSL_CERT_FILE on Nix-based CI) — clear them all so this test
    # asserts the production rule, not the runner's environment.
    for key in list(os.environ):
        if key in {"LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "LOCALE_ARCHIVE", "SSL_CERT_FILE"} or key.startswith(
            ("NIX_", "FONTCONFIG_", "AGENTIX_SAVED_")
        ):
            monkeypatch.delenv(key, raising=False)

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert not any(key.startswith("AGENTIX_SAVED_") for key in env)


def test_clean_worker_env_fresh_strip_overwrites_inherited_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested spawn: the freshly-observed live value is authoritative — an
    AGENTIX_SAVED_* snapshot inherited from an outer layer must not shadow it."""
    monkeypatch.setenv("AGENTIX_SAVED_PYTHONPATH", "/stale/outer")
    monkeypatch.setenv("PYTHONPATH", "/fresh/inner")

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert env["AGENTIX_SAVED_PYTHONPATH"] == "/fresh/inner"
    assert "PYTHONPATH" not in env
