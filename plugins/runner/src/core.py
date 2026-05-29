"""Batch rollout runner — the core library.

`run_rollouts(...)` runs an agent over a dataset of instances, each inside
its own sandbox, and returns a typed `Rollout` per instance. The agent
phase and the scoring phase each get a fresh sandbox — scoring must start
from a clean task image. Everything is built on the stable Agentix
surface: `provider.session(config)` yields a sandbox, and
`sandbox.remote(fn, ...)` runs a function inside it.

Datasets and agents are adapted through two small Protocols (`Dataset`,
`Agent`); the runner itself knows nothing benchmark- or agent-specific. An
RL/eval loop calls `run_rollouts(...)` directly — the `agentix-run` CLI is
a thin wrapper over the same function.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agentix import SandboxConfig

logger = logging.getLogger("agentix.runner")

__all__ = [
    "Agent",
    "AgentResult",
    "Dataset",
    "Provider",
    "Rollout",
    "rollout_one",
    "run_rollouts",
]


@dataclass(slots=True)
class AgentResult:
    """What an agent produced inside the sandbox: a patch (unified diff) plus
    optional metadata. `info` is free-form and the runner does not interpret
    it."""

    patch: str = ""
    exit_code: int | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Rollout:
    """The outcome of running one instance end-to-end.

    Exactly one of the terminal states holds: a `score` (instance reached the
    scorer), a `skipped` reason (`setup_failed` / `empty_patch`), or an
    `error` string (an unexpected exception, isolated to this instance)."""

    instance_id: str
    patch: str = ""
    score: dict[str, Any] | None = None
    agent_exit: int | None = None
    error: str | None = None
    skipped: str | None = None
    duration_s: float = 0.0

    @property
    def resolved(self) -> bool:
        """True when the scorer marked the instance resolved."""
        return bool(self.score and self.score.get("resolved"))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary — keeps the patch size, not its body."""
        return {
            "instance_id": self.instance_id,
            "resolved": self.resolved,
            "patch_bytes": len(self.patch),
            "agent_exit": self.agent_exit,
            "error": self.error,
            "skipped": self.skipped,
            "score": self.score,
            "duration_s": round(self.duration_s, 1),
        }


@runtime_checkable
class Dataset(Protocol):
    """Adapts a benchmark: enumerate instances, pick each task image, set the
    repo up inside the sandbox, and score a candidate patch.

    `setup` and `score` receive the live sandbox so they can issue
    `sandbox.remote(fn, ...)` calls (for example the
    `agentix.plugins.datasets.swe` `prepare_env` / `score` functions).
    `setup` returns whether preparation succeeded."""

    def instances(self) -> Iterable[dict[str, Any]]: ...

    def image(self, instance: dict[str, Any]) -> str: ...

    async def setup(self, sandbox: Any, instance: dict[str, Any]) -> bool: ...

    async def score(self, sandbox: Any, instance: dict[str, Any], patch: str) -> dict[str, Any]: ...


@runtime_checkable
class Agent(Protocol):
    """Adapts an agent: given a prepared sandbox and an instance, do the work
    and return the resulting patch. The adapter owns any agent-specific
    wiring (LLM bridge, model selection, patch extraction)."""

    async def solve(self, sandbox: Any, instance: dict[str, Any], *, model: str | None) -> AgentResult: ...


@runtime_checkable
class Provider(Protocol):
    """Anything that hands out a sandbox for a `SandboxConfig` as an async
    context manager — i.e. an Agentix `SandboxProvider`."""

    def session(self, config: SandboxConfig) -> AbstractAsyncContextManager[Any]: ...


def _instance_id(instance: dict[str, Any]) -> str:
    return str(instance.get("instance_id") or instance.get("id") or "?")


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def rollout_one(
    instance: dict[str, Any],
    *,
    dataset: Dataset,
    agent: Agent,
    provider: Provider,
    bundle: str,
    model: str | None = None,
    platform: str | None = None,
    score: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> Rollout:
    """Run a single instance: agent sandbox (setup then solve), then a fresh
    sandbox to score the patch. Task-level failures are recorded on the
    returned `Rollout`; this never raises for them."""
    iid = _instance_id(instance)
    started = clock()
    config = SandboxConfig(image=dataset.image(instance), bundle=bundle, platform=platform)

    try:
        async with provider.session(config) as sandbox:
            if not await dataset.setup(sandbox, instance):
                return Rollout(instance_id=iid, skipped="setup_failed", duration_s=clock() - started)
            result = await agent.solve(sandbox, instance, model=model)
    except Exception as exc:
        logger.exception("[%s] agent phase failed", iid)
        return Rollout(instance_id=iid, error=_error(exc), duration_s=clock() - started)

    if not result.patch.strip():
        return Rollout(
            instance_id=iid,
            agent_exit=result.exit_code,
            skipped="empty_patch",
            duration_s=clock() - started,
        )
    if not score:
        return Rollout(
            instance_id=iid,
            patch=result.patch,
            agent_exit=result.exit_code,
            duration_s=clock() - started,
        )

    try:
        async with provider.session(config) as sandbox:
            report = await dataset.score(sandbox, instance, result.patch)
    except Exception as exc:
        logger.exception("[%s] score phase failed", iid)
        return Rollout(
            instance_id=iid,
            patch=result.patch,
            agent_exit=result.exit_code,
            error=_error(exc),
            duration_s=clock() - started,
        )

    return Rollout(
        instance_id=iid,
        patch=result.patch,
        score=report,
        agent_exit=result.exit_code,
        duration_s=clock() - started,
    )


async def run_rollouts(
    *,
    dataset: Dataset,
    agent: Agent,
    provider: Provider,
    bundle: str,
    model: str | None = None,
    instances: Sequence[dict[str, Any]] | None = None,
    n_concurrent: int = 1,
    platform: str | None = None,
    score: bool = True,
    on_result: Callable[[Rollout], None] | None = None,
) -> list[Rollout]:
    """Run `agent` over every instance in `dataset` (or the explicit
    `instances`), at most `n_concurrent` at a time. Returns one `Rollout`
    per instance, in input order. Safe to call directly from an RL/eval
    loop — per-instance failures surface as `Rollout.error`, not
    exceptions. `on_result`, if given, fires as each instance finishes."""
    if n_concurrent < 1:
        raise ValueError("n_concurrent must be >= 1")

    items = list(instances) if instances is not None else list(dataset.instances())
    if not items:
        return []

    semaphore = asyncio.Semaphore(n_concurrent)

    async def _run(instance: dict[str, Any]) -> Rollout:
        async with semaphore:
            rollout = await rollout_one(
                instance,
                dataset=dataset,
                agent=agent,
                provider=provider,
                bundle=bundle,
                model=model,
                platform=platform,
                score=score,
            )
        if on_result is not None:
            on_result(rollout)
        return rollout

    return await asyncio.gather(*(_run(instance) for instance in items))
