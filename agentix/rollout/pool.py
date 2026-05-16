"""Parallel rollout pool: a fixed number of warm sandboxes shared across
many tasks. Bounds concurrency by sandbox slot, not by task count.

Typical use for an RL training loop:

    from agentix import DockerDeployment, RolloutPool, SandboxConfig
    from agentix.claude_code import ClaudeCode

    config = SandboxConfig(image="my-agent:0.1.0")

    async def rollout(client, task):
        return await client.remote(ClaudeCode.run, instruction=task)

    async with RolloutPool(DockerDeployment(), config, parallelism=16) as pool:
        async for task, result in pool.map(rollout, tasks):
            store_trajectory(task, result)

Trace events emitted from inside the namespaces during a rollout flow over
each sandbox's runtime Socket.IO. To correlate them, pass `call_id` when
making remote calls (RemoteRequest.call_id) — the dispatcher pins it into
a contextvar, and `agentix.trace.emit(...)` from the impl inherits it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Generic, TypeVar

from agentix.deployment.base import Deployment, Sandbox
from agentix.models import SandboxConfig
from agentix.runtime.client import RuntimeClient

logger = logging.getLogger("agentix.rollout")

T = TypeVar("T")
R = TypeVar("R")


class RolloutPool(Generic[T, R]):
    """A pool of `parallelism` warm sandboxes, each with its own RuntimeClient.

    Sandboxes are created in parallel on `__aenter__` and torn down on
    `__aexit__`. `map(fn, tasks)` runs `fn(client, task)` for each task,
    leasing a sandbox slot for the duration; slots are returned to the
    pool when the call finishes. Results are yielded as they complete
    (not in submission order).
    """

    def __init__(
        self,
        deployment: Deployment,
        config: SandboxConfig,
        parallelism: int = 8,
    ) -> None:
        if parallelism < 1:
            raise ValueError("parallelism must be >= 1")
        self._deployment = deployment
        self._config = config
        self._parallelism = parallelism
        self._sandboxes: list[Sandbox] = []
        self._clients: list[RuntimeClient] = []
        self._slot_queue: asyncio.Queue[tuple[Sandbox, RuntimeClient]] | None = None

    @property
    def parallelism(self) -> int:
        return self._parallelism

    @property
    def sandboxes(self) -> list[Sandbox]:
        """The pool's current sandbox handles (read-only)."""
        return list(self._sandboxes)

    async def __aenter__(self) -> RolloutPool[T, R]:
        logger.info("provisioning %d sandboxes in parallel", self._parallelism)
        self._sandboxes = await asyncio.gather(
            *(self._deployment.create(self._config) for _ in range(self._parallelism))
        )
        self._clients = [RuntimeClient(sb.runtime_url) for sb in self._sandboxes]
        # Open the queue with all slots free.
        self._slot_queue = asyncio.Queue()
        for slot in zip(self._sandboxes, self._clients):
            self._slot_queue.put_nowait(slot)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        # Close clients first so any in-flight Socket.IO disconnects gracefully,
        # then tear down sandboxes in parallel.
        await asyncio.gather(
            *(c.close() for c in self._clients),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(self._deployment.delete(sb.sandbox_id) for sb in self._sandboxes),
            return_exceptions=True,
        )
        self._sandboxes = []
        self._clients = []
        self._slot_queue = None

    async def map(
        self,
        fn: Callable[[RuntimeClient, T], Awaitable[R]],
        tasks: list[T] | tuple[T, ...],
    ) -> AsyncIterator[tuple[T, R | BaseException]]:
        """Run `fn(client, task)` for each task with bounded concurrency.

        Yields `(task, result)` pairs as they complete (NOT in submission
        order). If `fn` raises, the exception is yielded as the result —
        the pool keeps running so one bad task doesn't tank the rollout.
        """
        if self._slot_queue is None:
            raise RuntimeError("RolloutPool not entered (use `async with`)")

        if not tasks:
            return

        task_iter = iter(tasks)
        in_flight: set[asyncio.Task[tuple[T, R | BaseException]]] = set()

        async def _run_one(task: T) -> tuple[T, R | BaseException]:
            sb, client = await self._slot_queue.get()  # type: ignore[union-attr]
            try:
                try:
                    result: R | BaseException = await fn(client, task)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # noqa: BLE001 — surface to caller
                    logger.exception("rollout task failed")
                    result = exc
            finally:
                self._slot_queue.put_nowait((sb, client))  # type: ignore[union-attr]
            return task, result

        def _start_next() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            in_flight.add(asyncio.create_task(_run_one(task)))
            return True

        for _ in range(min(self._parallelism, len(tasks))):
            _start_next()

        try:
            while in_flight:
                done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    in_flight.remove(fut)
                    yield await fut
                    _start_next()
        finally:
            for fut in in_flight:
                fut.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)
