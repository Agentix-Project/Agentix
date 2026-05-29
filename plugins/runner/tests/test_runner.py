"""Tests for the batch rollout runner core.

These use an in-process fake provider whose sandbox runs `remote(fn, ...)`
calls locally, so the whole choreography is exercised without Docker.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from typing import Any

import pytest
from agentix.runner import AgentResult, run_rollouts


class _Sandbox:
    def __init__(self) -> None:
        self.namespaces: list[Any] = []

    async def remote(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    def register_namespace(self, namespace: Any) -> None:
        self.namespaces.append(namespace)


class _Provider:
    def __init__(self) -> None:
        self.sessions = 0
        self.active = 0
        self.max_active = 0

    @asynccontextmanager
    async def session(self, config: Any) -> Any:
        self.sessions += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            yield _Sandbox()
        finally:
            self.active -= 1


class _Dataset:
    def __init__(self, instances: list[dict[str, Any]]) -> None:
        self._instances = instances

    def instances(self) -> list[dict[str, Any]]:
        return list(self._instances)

    def image(self, instance: dict[str, Any]) -> str:
        return f"task:{instance['instance_id']}"

    async def setup(self, sandbox: Any, instance: dict[str, Any]) -> bool:
        await sandbox.remote(lambda: None)
        return not instance.get("setup_fails", False)

    async def score(self, sandbox: Any, instance: dict[str, Any], patch: str) -> dict[str, Any]:
        return {"resolved": patch == instance.get("gold", "GOLD"), "patch_applied": True}


class _OracleAgent:
    async def solve(self, sandbox: Any, instance: dict[str, Any], *, model: str | None) -> AgentResult:
        delay = float(instance.get("delay", 0.0))
        if delay:
            await asyncio.sleep(delay)
        if instance.get("agent_raises"):
            raise RuntimeError("boom")
        return AgentResult(patch=instance.get("produce", "GOLD"), exit_code=0)


def _inst(iid: str, **extra: Any) -> dict[str, Any]:
    return {"instance_id": iid, **extra}


async def test_resolved_rollout_uses_two_sandboxes() -> None:
    provider = _Provider()
    [rollout] = await run_rollouts(
        dataset=_Dataset([_inst("a")]),
        agent=_OracleAgent(),
        provider=provider,
        bundle="b:0",
    )
    assert rollout.resolved
    assert rollout.agent_exit == 0
    assert rollout.score == {"resolved": True, "patch_applied": True}
    assert provider.sessions == 2


async def test_empty_patch_skips_scoring() -> None:
    provider = _Provider()
    [rollout] = await run_rollouts(
        dataset=_Dataset([_inst("a", produce="")]),
        agent=_OracleAgent(),
        provider=provider,
        bundle="b:0",
    )
    assert rollout.skipped == "empty_patch"
    assert rollout.score is None
    assert provider.sessions == 1


async def test_setup_failure_is_recorded() -> None:
    provider = _Provider()
    [rollout] = await run_rollouts(
        dataset=_Dataset([_inst("a", setup_fails=True)]),
        agent=_OracleAgent(),
        provider=provider,
        bundle="b:0",
    )
    assert rollout.skipped == "setup_failed"
    assert provider.sessions == 1


async def test_agent_crash_is_isolated() -> None:
    rollouts = await run_rollouts(
        dataset=_Dataset([_inst("bad", agent_raises=True), _inst("good")]),
        agent=_OracleAgent(),
        provider=_Provider(),
        bundle="b:0",
    )
    by_id = {r.instance_id: r for r in rollouts}
    assert by_id["bad"].error is not None
    assert "boom" in by_id["bad"].error
    assert by_id["good"].resolved


async def test_no_score_returns_patch_without_scoring() -> None:
    provider = _Provider()
    [rollout] = await run_rollouts(
        dataset=_Dataset([_inst("a")]),
        agent=_OracleAgent(),
        provider=provider,
        bundle="b:0",
        score=False,
    )
    assert rollout.patch == "GOLD"
    assert rollout.score is None
    assert provider.sessions == 1


async def test_concurrency_is_bounded_and_order_preserved() -> None:
    provider = _Provider()
    instances = [_inst(str(i), delay=0.02 if i % 2 == 0 else 0.0) for i in range(8)]
    rollouts = await run_rollouts(
        dataset=_Dataset(instances),
        agent=_OracleAgent(),
        provider=provider,
        bundle="b:0",
        n_concurrent=3,
    )
    assert [r.instance_id for r in rollouts] == [str(i) for i in range(8)]
    assert provider.max_active <= 3


async def test_on_result_called_per_instance() -> None:
    seen: list[str] = []
    await run_rollouts(
        dataset=_Dataset([_inst("a"), _inst("b")]),
        agent=_OracleAgent(),
        provider=_Provider(),
        bundle="b:0",
        on_result=lambda rollout: seen.append(rollout.instance_id),
    )
    assert sorted(seen) == ["a", "b"]


async def test_empty_dataset_returns_empty() -> None:
    rollouts = await run_rollouts(
        dataset=_Dataset([]),
        agent=_OracleAgent(),
        provider=_Provider(),
        bundle="b:0",
    )
    assert rollouts == []


async def test_invalid_concurrency_raises() -> None:
    with pytest.raises(ValueError, match="n_concurrent"):
        await run_rollouts(
            dataset=_Dataset([_inst("a")]),
            agent=_OracleAgent(),
            provider=_Provider(),
            bundle="b:0",
            n_concurrent=0,
        )
