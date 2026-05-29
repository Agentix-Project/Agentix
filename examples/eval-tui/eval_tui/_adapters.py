"""Phase-tracing wrappers around a runner `Dataset` / `Agent`.

The runner exposes a per-instance `on_result` callback but no in-flight phase
hook. To drive a live UI we wrap the dataset/agent so the dashboard learns
when each instance enters `setup`, `agent`, and `score` — without changing
`agentix.runner` itself. Each wrapper simply emits a phase event, then
delegates to the wrapped object.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

OnPhase = Callable[[str, str], None]


def instance_id(instance: dict[str, Any]) -> str:
    return str(instance.get("instance_id") or instance.get("id") or "?")


class TracingDataset:
    def __init__(self, inner: Any, on_phase: OnPhase) -> None:
        self._inner = inner
        self._on_phase = on_phase

    def instances(self) -> Any:
        return self._inner.instances()

    def image(self, instance: dict[str, Any]) -> str:
        return self._inner.image(instance)

    async def setup(self, sandbox: Any, instance: dict[str, Any]) -> bool:
        self._on_phase(instance_id(instance), "setup")
        return await self._inner.setup(sandbox, instance)

    async def score(self, sandbox: Any, instance: dict[str, Any], patch: str) -> dict[str, Any]:
        self._on_phase(instance_id(instance), "score")
        return await self._inner.score(sandbox, instance, patch)


class TracingAgent:
    def __init__(self, inner: Any, on_phase: OnPhase) -> None:
        self._inner = inner
        self._on_phase = on_phase

    async def solve(self, sandbox: Any, instance: dict[str, Any], *, model: str | None) -> Any:
        self._on_phase(instance_id(instance), "agent")
        return await self._inner.solve(sandbox, instance, model=model)
