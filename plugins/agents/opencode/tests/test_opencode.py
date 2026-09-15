"""Tests for the opencode agent wrapper — argv/env construction and timeout,
exercised without invoking the real `opencode` binary.
"""

from __future__ import annotations

import asyncio
from typing import Any

import agentix.agents.opencode as opencode


class _FakeProc:
    def __init__(self, *, returncode: int = 0, out: bytes = b"done", err: bytes = b"", hang: bool = False) -> None:
        self.returncode = returncode
        self._out = out
        self._err = err
        self._hang = hang
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang and not self.killed:
            await asyncio.sleep(10)
        return self._out, self._err

    def kill(self) -> None:
        self.killed = True


async def test_run_builds_argv_and_env(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    async def _fake(*cmd: str, **kwargs: Any) -> _FakeProc:
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return _FakeProc(out=b"edited")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake)

    result = await opencode.run(
        "fix the bug",
        workdir="/repo",
        model="openai/gpt-4o",
        extra_args=["--agent", "build"],
        env={"OPENAI_BASE_URL": "http://bridge", "OPENAI_API_KEY": "k"},
    )

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-4o"
    assert cmd[-1] == "fix the bug"
    assert "--agent" in cmd
    assert captured["kwargs"]["cwd"] == "/repo"
    assert captured["kwargs"]["env"]["OPENAI_BASE_URL"] == "http://bridge"
    assert result.exit_code == 0
    assert result.stdout == "edited"


async def test_run_times_out_and_kills(monkeypatch: Any) -> None:
    proc = _FakeProc(hang=True)

    async def _fake(*cmd: str, **kwargs: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake)

    result = await opencode.run("x", timeout=0.01)
    assert result.exit_code == -1
    assert "timed out" in result.stderr
    assert proc.killed
