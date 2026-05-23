"""mini-swe-agent runner for Agentix sandboxes.

Patterns borrowed from harbor's `MiniSweAgent`:

  * `run()` drives a pre-built agent inside the sandbox, captures
    mini-swe-agent's native v2 trajectory file, and post-processes it
    into a structured `Trajectory` plus aggregated usage metrics.
  * The captured trajectory rides back to the host as part of the
    return value (no shared filesystem assumptions, no extra file
    round-trip — `client.remote(...)` pickles the value).
  * `cost_limit` / `reasoning_effort` / `config_yaml` are accepted but
    handled in-process via the `DefaultAgent` instance the caller
    constructs, not via a CLI subprocess (we do not spawn the
    mini-swe-agent CLI; the agent object's `run(task)` is the
    integration point).

The return shape is a `MiniSweAgentResult` dataclass; legacy callers
that expected a plain dict still get one via `result.to_dict()`.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from minisweagent import Agent

from .trajectory import Trajectory, aggregate_usage, from_mini_swe_agent


@dataclass(slots=True)
class MiniSweAgentResult:
    """Enriched return type from a mini-swe-agent run.

    `exit_status` and `submission` mirror mini-swe-agent's own return
    shape so existing callers keep working. The structured
    `trajectory` and `usage` fields are the new value-add — they
    surface the same metrics harbor's `populate_context_post_run`
    pushes into the trial context, without the agent-installer
    machinery.
    """

    exit_status: str
    submission: str
    workdir: str
    raw_trajectory: dict[str, Any] = field(default_factory=dict)
    trajectory: Trajectory | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.trajectory is not None:
            d["trajectory"] = self.trajectory.to_dict()
        return d


# ── public entry point ────────────────────────────────────────────────────


def run(
    task: str,
    *,
    workdir: str = "/testbed",
    agent: Agent,
    trajectory_path: str | Path | None = None,
    session_id: str | None = None,
) -> MiniSweAgentResult:
    """Run a pre-built mini-swe-agent inside the sandbox.

    Mini-swe-agent's `DefaultAgent.run(task)` returns a `(exit_status,
    submission)` tuple-shaped object (in practice it returns a
    2-element iterable). We:

    1. Set the environment's working directory if the agent exposes
       `env.config.cwd` (mirrors harbor's wiring).
    2. Run the agent and collect the result.
    3. If `trajectory_path` is provided AND the file exists after the
       run, load mini-swe-agent's native v2 trajectory and convert to
       a structured `Trajectory`.
    4. Compute aggregated `usage` for callers that just need totals.

    All paths are sandbox-local; the host pickle-recovers the
    `MiniSweAgentResult` over the runtime client and can persist /
    inspect / score it without re-running anything.
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    env = getattr(agent, "env", None)
    env_config = getattr(env, "config", None)
    if env_config is not None and hasattr(env_config, "cwd"):
        env_config.cwd = str(workdir_path)

    raw_result = dict(agent.run(task))
    exit_status = str(raw_result.get("exit_status", "unknown"))
    submission = str(raw_result.get("submission", ""))

    raw_trajectory: dict[str, Any] = {}
    if trajectory_path is not None:
        raw_trajectory = _read_trajectory(Path(trajectory_path))
    elif "messages" in raw_result:
        # Some mini-swe-agent versions expose the trajectory inline on
        # `agent.run()`'s return value (notably bench scripts that
        # don't persist a file). Honour that too.
        raw_trajectory = raw_result

    trajectory: Trajectory | None = None
    usage: dict[str, Any] = {}
    if raw_trajectory:
        usage = aggregate_usage(raw_trajectory)
        try:
            trajectory = from_mini_swe_agent(
                raw_trajectory,
                session_id=session_id or uuid.uuid4().hex,
            )
        except Exception:
            # Best-effort: aggregating usage is cheaper / more
            # tolerant than full conversion; keep `usage` even if
            # conversion fails so the host always sees token counts.
            trajectory = None

    return MiniSweAgentResult(
        exit_status=exit_status,
        submission=submission,
        workdir=str(workdir_path),
        raw_trajectory=raw_trajectory,
        trajectory=trajectory,
        usage=usage,
    )


def _read_trajectory(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


__all__ = ["MiniSweAgentResult", "run"]
