"""Tests for the mini-swe-agent runner and trajectory conversion.

Covers:

  * `run(...)` returns a `MiniSweAgentResult` carrying `exit_status`
    and `submission` from the agent, the resolved `workdir`, and the
    raw trajectory dict.
  * When a `trajectory_path` is provided, the structured
    `Trajectory` is constructed from mini-swe-agent's v2 format,
    including system / user / assistant / tool / tool_use steps.
  * `aggregate_usage(...)` matches what `from_mini_swe_agent(...)`
    produces in `final_metrics` (consistency check between the
    cheap and full paths).
  * Errors inside the agent surface as exceptions instead of being
    swallowed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import agentix.agents.mini_swe_agent as mini_swe
import pytest


class DummyEnvConfig:
    def __init__(self) -> None:
        self.cwd = ""


class DummyEnv:
    def __init__(self) -> None:
        self.config = DummyEnvConfig()


def _v2_trajectory() -> dict[str, Any]:
    """Minimal mini-swe-agent v2 trajectory with system/user/assistant/tool."""
    usage_a = {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 1},
    }
    usage_b = {"prompt_tokens": 20, "completion_tokens": 6}
    return {
        "trajectory_format": "mini-swe-agent.v2",
        "info": {
            "mini_version": "2.3.0",
            "config": {
                "model": {"model_name": "openai/gpt-4o-mini"},
                "agent": {"mode": "yolo"},
            },
            "model_stats": {"instance_cost": 0.012},
        },
        "messages": [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": "thinking...",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
                    }
                ],
                "extra": {"response": {"usage": usage_a}},
            },
            {"role": "tool", "content": "file_a.py\nfile_b.py"},
            {
                "role": "assistant",
                "content": "done",
                "extra": {"response": {"usage": usage_b}},
            },
        ],
    }


# ── run() ─────────────────────────────────────────────────────────────────


def test_run_returns_structured_result(tmp_path: Path) -> None:
    class DummyAgent:
        def __init__(self) -> None:
            self.env = DummyEnv()

        def run(self, _: str):
            return {"exit_status": "submitted", "submission": "diff --git ..."}

    agent = DummyAgent()
    result = mini_swe.run("fix bug", workdir=str(tmp_path), agent=agent)

    assert isinstance(result, mini_swe.MiniSweAgentResult)
    assert result.exit_status == "submitted"
    assert result.submission == "diff --git ..."
    assert result.workdir == str(tmp_path)
    assert agent.env.config.cwd == str(tmp_path)
    # No trajectory_path passed and `run` returned a plain result
    # without `messages` -> no trajectory.
    assert result.trajectory is None
    assert result.usage == {}


def test_run_loads_trajectory_from_file(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "mini-swe-agent.trajectory.json"
    trajectory_path.write_text(json.dumps(_v2_trajectory()))

    class DummyAgent:
        def __init__(self) -> None:
            self.env = DummyEnv()

        def run(self, _: str):
            return {"exit_status": "submitted", "submission": "patch"}

    result = mini_swe.run(
        "fix",
        workdir=str(tmp_path),
        agent=DummyAgent(),
        trajectory_path=trajectory_path,
        session_id="sess-test",
    )

    assert result.trajectory is not None
    traj = result.trajectory
    assert traj.session_id == "sess-test"
    assert traj.agent.name == "mini-swe-agent"
    assert traj.agent.version == "2.3.0"
    assert traj.agent.model_name == "openai/gpt-4o-mini"

    sources = [s.source for s in traj.steps]
    assert sources == ["system", "user", "agent", "agent"]
    # First assistant carries a tool call and a tool_result observation
    # was attached to that same step.
    [first_agent, second_agent] = [s for s in traj.steps if s.source == "agent"]
    assert first_agent.tool_calls is not None
    assert first_agent.tool_calls[0].function_name == "bash"
    assert first_agent.tool_calls[0].arguments == {"command": "ls"}
    assert first_agent.observation is not None
    assert first_agent.observation.results[0].content == "file_a.py\nfile_b.py"
    # Second assistant — no tool call, message text preserved.
    assert second_agent.tool_calls is None
    assert second_agent.message == "done"

    # Final metrics aggregate correctly.
    assert traj.final_metrics.total_prompt_tokens == 32
    assert traj.final_metrics.total_completion_tokens == 10
    assert traj.final_metrics.total_cached_tokens == 3
    assert traj.final_metrics.total_cost_usd == 0.012
    assert (traj.final_metrics.extra or {}).get("total_reasoning_tokens") == 1

    # Cheap aggregate matches the full path.
    assert result.usage == {
        "n_input_tokens": 32,
        "n_output_tokens": 10,
        "n_cache_tokens": 3,
        "cost_usd": 0.012,
    }


def test_run_inline_trajectory_passthrough(tmp_path: Path) -> None:
    """Some bench scripts return the trajectory inline. Honour that path."""
    inline = _v2_trajectory()
    inline["exit_status"] = "submitted"
    inline["submission"] = "patch"

    class DummyAgent:
        def __init__(self) -> None:
            self.env = DummyEnv()

        def run(self, _: str):
            return inline

    result = mini_swe.run("fix", workdir=str(tmp_path), agent=DummyAgent())
    assert result.trajectory is not None
    assert result.usage["n_input_tokens"] == 32


def test_run_exception_propagates(tmp_path: Path) -> None:
    class BoomAgent:
        def __init__(self) -> None:
            self.env = DummyEnv()

        def run(self, _: str):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mini_swe.run("fix", workdir=str(tmp_path), agent=BoomAgent())


# ── trajectory module direct tests ────────────────────────────────────────


def test_aggregate_usage_matches_final_metrics() -> None:
    raw = _v2_trajectory()
    usage = mini_swe.aggregate_usage(raw)
    traj = mini_swe.from_mini_swe_agent(raw, session_id="sid")
    assert traj.final_metrics.total_prompt_tokens == usage["n_input_tokens"]
    assert traj.final_metrics.total_completion_tokens == usage["n_output_tokens"]


def test_trajectory_to_dict_strips_none() -> None:
    raw = _v2_trajectory()
    traj = mini_swe.from_mini_swe_agent(raw, session_id="sid")
    d = traj.to_dict()
    # No top-level None values
    for v in d.values():
        if isinstance(v, dict):
            assert all(value is not None for value in v.values())
    # SCHEMA_VERSION exposed for downstream consumers.
    assert d["schema_version"] == mini_swe.SCHEMA_VERSION


def test_trajectory_to_json_is_valid() -> None:
    raw = _v2_trajectory()
    traj = mini_swe.from_mini_swe_agent(raw, session_id="sid")
    parsed = json.loads(traj.to_json())
    assert parsed["agent"]["name"] == "mini-swe-agent"
    assert parsed["steps"][0]["source"] == "system"


def test_tool_call_with_string_arguments_falls_back_to_command() -> None:
    raw: dict[str, Any] = {
        "trajectory_format": "mini-swe-agent.v2",
        "info": {"mini_version": "2.3", "model_stats": {"instance_cost": 0.0}},
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "bash", "arguments": "ls -la"}}
                ],
                "extra": {"response": {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}},
            },
        ],
    }
    traj = mini_swe.from_mini_swe_agent(raw, session_id="sid")
    [agent_step] = [s for s in traj.steps if s.source == "agent"]
    assert agent_step.tool_calls is not None
    assert agent_step.tool_calls[0].arguments == {"command": "ls -la"}
