"""Sandbox-side mini-swe-agent runner.

This integration intentionally uses mini-swe-agent's Python API directly
instead of shelling out to the `mini` CLI.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Result:
    exit_status: str
    submission: str
    output_path: str | None
    cost: float
    n_calls: int
    details: dict[str, Any]


@dataclass
class _SyncResult:
    details: dict[str, Any]
    output_path: str | None
    cost: float
    n_calls: int


async def run(
    task: str,
    *,
    workdir: str = "/testbed",
    model: str | None = None,
    timeout: float = 1800,
    max_steps: int | None = None,
    output_path: str | None = None,
    env: dict[str, str] | None = None,
) -> Result:
    """Run mini-swe-agent in-process with a local environment."""
    try:
        sync_result = await asyncio.wait_for(
            asyncio.to_thread(
                _run_sync,
                task=task,
                workdir=workdir,
                model=model,
                max_steps=max_steps,
                output_path=output_path,
                env=env or {},
            ),
            timeout=timeout,
        )
    except TimeoutError:
        return Result(
            exit_status="timeout",
            submission="",
            output_path=output_path,
            cost=0.0,
            n_calls=0,
            details={"error": f"mini-swe-agent timed out after {timeout}s"},
        )
    except Exception as exc:
        return Result(
            exit_status="error",
            submission="",
            output_path=output_path,
            cost=0.0,
            n_calls=0,
            details={"error": f"{type(exc).__name__}: {exc}"},
        )

    details = sync_result.details
    return Result(
        exit_status=str(details.get("exit_status", "unknown")),
        submission=str(details.get("submission", "")),
        output_path=sync_result.output_path,
        cost=sync_result.cost,
        n_calls=sync_result.n_calls,
        details=details,
    )


def _run_sync(
    *,
    task: str,
    workdir: str,
    model: str | None,
    max_steps: int | None,
    output_path: str | None,
    env: dict[str, str],
) -> _SyncResult:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models import get_model

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    run_output_path = output_path or str(workdir_path / ".mini-swe-agent.jsonl")

    model_cfg: dict[str, Any] = {}
    if model:
        model_cfg["model_name"] = model

    with _pushd(workdir_path), _patched_env(env):
        model_obj = get_model(config=model_cfg)
        env_obj = LocalEnvironment()
        kwargs: dict[str, Any] = {"output_path": run_output_path}
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        agent = DefaultAgent(model_obj, env_obj, **kwargs)
        details = agent.run(task)

    return _SyncResult(
        details=details,
        output_path=run_output_path,
        cost=float(getattr(agent, "cost", 0.0)),
        n_calls=int(getattr(agent, "n_calls", 0)),
    )


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def _patched_env(extra: dict[str, str]):
    old_env: dict[str, str | None] = {}
    for key, value in extra.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


__all__ = ["Result", "run"]
