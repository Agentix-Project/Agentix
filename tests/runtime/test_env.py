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


def test_get_env_without_agentix_restores_saved_vars() -> None:
    """Vars the worker spawn STRIPPED (PYTHONPATH, LD_PRELOAD, ...) are
    recorded under AGENTIX_SAVED_*; the clean env restores them so task
    commands see the image's original environment."""
    env = get_env_without_agentix(
        base={
            "AGENTIX_SAVED_PYTHONPATH": "/testbed/src",
            "AGENTIX_SAVED_LD_PRELOAD": "/usr/lib/libfoo.so",
            "PATH": "/usr/bin",
        }
    )

    assert env["PYTHONPATH"] == "/testbed/src"
    assert env["LD_PRELOAD"] == "/usr/lib/libfoo.so"
    assert "AGENTIX_SAVED_PYTHONPATH" not in env
    assert "AGENTIX_SAVED_LD_PRELOAD" not in env


def test_get_env_without_agentix_does_not_clobber_live_values_on_restore() -> None:
    """A live value is more recent intent than the pre-strip snapshot."""
    env = get_env_without_agentix(
        base={
            "PYTHONPATH": "/set/after/spawn",
            "AGENTIX_SAVED_PYTHONPATH": "/original",
        }
    )

    assert env["PYTHONPATH"] == "/set/after/spawn"
    assert "AGENTIX_SAVED_PYTHONPATH" not in env


def test_get_env_without_agentix_extra_wins_over_restored() -> None:
    env = get_env_without_agentix(
        {"PYTHONPATH": "/caller"},
        base={"AGENTIX_SAVED_PYTHONPATH": "/original"},
    )

    assert env["PYTHONPATH"] == "/caller"


def test_bash_clean_env_opt_in_builds_the_image_environment(monkeypatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["/nix/runtime/bin", "/task/bin"]))
    monkeypatch.setenv(AGENTIX_ADDED_PATH, "/nix/runtime/bin")
    monkeypatch.setenv("AGENTIX_SAVED_PYTHONPATH", "/testbed/src")

    env = _clean_env({"EXTRA": "1"}, clean=True)

    assert env["PATH"] == "/task/bin"
    assert env["PYTHONPATH"] == "/testbed/src"
    assert AGENTIX_ADDED_PATH not in env
    assert env["EXTRA"] == "1"
