from __future__ import annotations

import os

from agentix.bash import _clean_env

from agentix.runtime.shared.env import (
    AGENTIX_ADDED_LD_LIBRARY_PATH,
    AGENTIX_ADDED_PATH,
    get_env_without_agentix,
)


def test_get_env_without_agentix_removes_only_recorded_path_entries() -> None:
    env = get_env_without_agentix(
        base={
            "PATH": os.pathsep.join(
                [
                    "/nix/runtime/venv/bin",
                    "/task/nix/bin",
                    "/usr/bin",
                    "/nix/runtime/bin",
                ]
            ),
            AGENTIX_ADDED_PATH: os.pathsep.join(
                [
                    "/nix/runtime/venv/bin",
                    "/nix/runtime/bin",
                ]
            ),
            "TASK_MARKER": "kept",
        }
    )

    assert env["PATH"].split(os.pathsep) == ["/task/nix/bin", "/usr/bin"]
    assert env["TASK_MARKER"] == "kept"
    assert AGENTIX_ADDED_PATH not in env


def test_get_env_without_agentix_removes_recorded_ld_library_entries() -> None:
    env = get_env_without_agentix(
        base={
            "LD_LIBRARY_PATH": os.pathsep.join(
                [
                    "/nix/runtime/lib",
                    "/task/lib",
                    "/another/task/lib",
                ]
            ),
            AGENTIX_ADDED_LD_LIBRARY_PATH: "/nix/runtime/lib",
        }
    )

    assert env["LD_LIBRARY_PATH"].split(os.pathsep) == ["/task/lib", "/another/task/lib"]
    assert AGENTIX_ADDED_LD_LIBRARY_PATH not in env


def test_get_env_without_agentix_removes_arbitrary_recorded_path_vars() -> None:
    env = get_env_without_agentix(
        base={
            "PKG_CONFIG_PATH": os.pathsep.join(
                [
                    "/nix/runtime/lib/pkgconfig",
                    "/task/lib/pkgconfig",
                ]
            ),
            "AGENTIX_ADDED_PKG_CONFIG_PATH": "/nix/runtime/lib/pkgconfig",
        }
    )

    assert env["PKG_CONFIG_PATH"] == "/task/lib/pkgconfig"
    assert "AGENTIX_ADDED_PKG_CONFIG_PATH" not in env


def test_get_env_without_agentix_applies_extra_last() -> None:
    env = get_env_without_agentix(
        {"PATH": "/override/bin", AGENTIX_ADDED_PATH: "/caller/value"},
        base={
            "PATH": os.pathsep.join(["/nix/runtime/bin", "/usr/bin"]),
            AGENTIX_ADDED_PATH: "/nix/runtime/bin",
        },
    )

    assert env["PATH"] == "/override/bin"
    assert env[AGENTIX_ADDED_PATH] == "/caller/value"


def test_bash_clean_env_inherits_agentix_runtime_env(monkeypatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["/nix/runtime/bin", "/task/bin"]))
    monkeypatch.setenv(AGENTIX_ADDED_PATH, "/nix/runtime/bin")

    env = _clean_env(None)

    assert env["PATH"].split(os.pathsep) == ["/nix/runtime/bin", "/task/bin"]
    assert env[AGENTIX_ADDED_PATH] == "/nix/runtime/bin"
