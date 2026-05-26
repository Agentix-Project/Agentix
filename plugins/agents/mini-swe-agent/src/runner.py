from __future__ import annotations

from pathlib import Path

from minisweagent import Agent


def run(
    task: str,
    *,
    workdir: str = "/testbed",
    agent: Agent,
) -> None:
    """Run a pre-built mini-swe-agent instance in the sandbox.

    Trajectory capture is left entirely to abridge at the proxy layer;
    the integration only invokes the agent.
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    env = getattr(agent, "env", None)
    env_config = getattr(env, "config", None)
    if env_config is not None and hasattr(env_config, "cwd"):
        env_config.cwd = str(workdir_path)

    agent.run(task)
