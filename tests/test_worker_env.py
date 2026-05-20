from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentix.runtime.server.worker.client import _RUNTIME_BIN_PATH, _clean_worker_env


def test_clean_worker_env_inherits_parent_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["/custom/bin", "/usr/bin"]))

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert env["PATH"].split(os.pathsep) == [
        "/runtime/venv/bin",
        _RUNTIME_BIN_PATH,
        "/custom/bin",
        "/usr/bin",
    ]


def test_clean_worker_env_dedupes_prepended_path_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([_RUNTIME_BIN_PATH, "/usr/bin", "/runtime/venv/bin"]),
    )

    env = _clean_worker_env(Path("/runtime/venv/bin"))

    assert env["PATH"].split(os.pathsep) == [
        "/runtime/venv/bin",
        _RUNTIME_BIN_PATH,
        "/usr/bin",
    ]
