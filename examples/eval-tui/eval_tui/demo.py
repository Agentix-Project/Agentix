"""Synthetic, no-Docker dataset/agent/provider for the dashboard demo.

Outcomes are pre-decided from a seed so a demo run is reproducible. Phase
durations are scaled by `dur_scale` (tests use a tiny value to run fast).
"""

from __future__ import annotations

import asyncio
import random
from contextlib import asynccontextmanager
from typing import Any

from agentix.runner import AgentResult


class DemoProvider:
    @asynccontextmanager
    async def session(self, config: Any) -> Any:
        yield _DemoSandbox()


class _DemoSandbox:
    async def remote(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    def register_namespace(self, namespace: Any) -> None:
        pass


class DemoDataset:
    """`n` synthetic instances with seeded, pre-decided outcomes."""

    def __init__(self, n: int, *, seed: int = 0, dur_scale: float = 1.0) -> None:
        rng = random.Random(seed)
        self._rows: list[dict[str, Any]] = []
        for i in range(max(0, n)):
            self._rows.append(
                {
                    "instance_id": f"demo__task-{i:03d}",
                    "_setup_ok": rng.random() > 0.04,
                    "_solvable": rng.random() > 0.4,
                    "_empty": rng.random() < 0.12,
                    "_raises": rng.random() < 0.05,
                    "_dur": rng.uniform(0.3, 1.6) * dur_scale,
                }
            )

    def instances(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]

    def image(self, instance: dict[str, Any]) -> str:
        return "demo:latest"

    async def setup(self, sandbox: Any, instance: dict[str, Any]) -> bool:
        await asyncio.sleep(float(instance["_dur"]) * 0.25)
        return bool(instance["_setup_ok"])

    async def score(self, sandbox: Any, instance: dict[str, Any], patch: str) -> dict[str, Any]:
        await asyncio.sleep(float(instance["_dur"]) * 0.25)
        return {"resolved": bool(instance["_solvable"]), "patch_applied": True}


class DemoAgent:
    async def solve(self, sandbox: Any, instance: dict[str, Any], *, model: str | None) -> AgentResult:
        await asyncio.sleep(float(instance["_dur"]))
        if instance.get("_raises"):
            raise RuntimeError("synthetic agent failure")
        if instance.get("_empty"):
            return AgentResult(patch="", exit_code=0)
        return AgentResult(patch=f"--- diff for {instance['instance_id']} ---\n", exit_code=0)
