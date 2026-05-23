from __future__ import annotations

import agentix.agents.mini_swe_agent as mini_swe
import pytest


class DummyEnvConfig:
    def __init__(self) -> None:
        self.cwd = ""


class DummyEnv:
    def __init__(self) -> None:
        self.config = DummyEnvConfig()


def test_run_invokes_agent(tmp_path):
    class DummyAgent:
        def __init__(self) -> None:
            self.env = DummyEnv()
            self.called_with: str | None = None

        def run(self, task: str):
            self.called_with = task
            return {"exit_status": "submitted", "submission": "diff --git ..."}

    agent = DummyAgent()
    result = mini_swe.run(
        "fix bug",
        workdir=str(tmp_path),
        agent=agent,
    )
    assert result is None
    assert agent.called_with == "fix bug"
    assert agent.env.config.cwd == str(tmp_path)


def test_run_exception_propagates(tmp_path):
    class BoomAgent:
        def __init__(self) -> None:
            self.env = DummyEnv()

        def run(self, task: str):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mini_swe.run(
            "fix bug",
            workdir=str(tmp_path),
            agent=BoomAgent(),
        )
