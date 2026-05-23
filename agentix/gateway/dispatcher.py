"""Stage orchestration for a single gateway session.

Each session runs through four ordered stages — INIT, READY, RUNNING,
POSTRUN — with a single-worker-per-session executor backed by
`asyncio.Task`. Agentix's runtime already isolates per-sandbox state,
so we don't need a separate process pool per stage.

Stages:

  INIT      provision the sandbox via `Deployment.create(...)`;
            stand up the runtime client.
  READY     attach the abridge `OpenAICompatibleClient` so the
            sandbox's LLM traffic flows through the host's upstream.
  RUNNING   resolve the agent callable and `remote()`-invoke it.
            Pause is a no-op until generation is in flight.
  POSTRUN   tear down the sandbox + finalise the session result.

`Dispatcher.dispatch(spec)` returns the `Session` once it's queued;
the orchestration runs in the background. Callers poll the
`SessionStore` or register a result callback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any

from agentix import RuntimeClient
from agentix.deployment.base import Deployment, SandboxConfig, SandboxId
from agentix.gateway.session import Session, SessionResult, SessionSpec, SessionStatus
from agentix.gateway.storage import RecordStore, SessionStore

logger = logging.getLogger("agentix.gateway.dispatcher")

ResultCallback = Callable[[SessionResult], Awaitable[None] | None]


class DispatchStage:
    """Stage names exposed for tracing / logging consumers."""

    INIT = "init"
    READY = "ready"
    RUNNING = "running"
    POSTRUN = "postrun"


class Dispatcher:
    """Orchestrates one session through INIT -> READY -> RUNNING -> POSTRUN.

    Construct once per gateway node; call `dispatch(spec)` per
    incoming session. The dispatcher owns concurrency: a semaphore
    bounds in-flight sessions so a coordinator can't stampede a
    single node.
    """

    def __init__(
        self,
        *,
        deployment: Deployment,
        host_namespace_factory: Callable[[Session], Any] | None = None,
        sessions: SessionStore | None = None,
        records: RecordStore | None = None,
        concurrency: int = 4,
        result_callback: ResultCallback | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._deployment = deployment
        self._host_namespace_factory = host_namespace_factory
        self._sessions = sessions if sessions is not None else SessionStore()
        self._records = records if records is not None else RecordStore()
        self._semaphore = asyncio.Semaphore(concurrency)
        self._result_callback = result_callback
        self._paused = asyncio.Event()
        self._paused.set()  # set => not paused
        self._tasks: dict[str, asyncio.Task] = {}

    @property
    def sessions(self) -> SessionStore:
        return self._sessions

    @property
    def records(self) -> RecordStore:
        return self._records

    @property
    def paused(self) -> bool:
        return not self._paused.is_set()

    # ── pause / resume controls ──────────────────────────────────────

    def pause(self) -> None:
        """Block new RUNNING-stage transitions.

        Existing RUNNING sessions are not interrupted (would require
        cancellation in the sandbox); future sessions wait at the
        boundary between READY and RUNNING.
        """
        self._paused.clear()

    def resume(self) -> None:
        self._paused.set()

    # ── dispatch ─────────────────────────────────────────────────────

    def dispatch(self, spec: SessionSpec) -> Session:
        session = Session(spec=spec)
        if spec.session_id:
            session.session_id = spec.session_id
        self._sessions.register(session)
        task = asyncio.create_task(self._run(session))
        self._tasks[session.session_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(session.session_id, None))
        return session

    async def _run(self, session: Session) -> None:
        async with self._semaphore:
            try:
                await self._stage_init(session)
                await self._stage_ready(session)
                await self._paused.wait()
                await self._stage_running(session)
                session.mark(SessionStatus.POSTRUN)
                await self._stage_postrun(session)
                session.mark(SessionStatus.SUCCEEDED)
            except asyncio.CancelledError:
                session.mark(SessionStatus.CANCELLED, error="cancelled")
                raise
            except Exception as exc:  # noqa: BLE001 - surface any failure into the result
                session.mark(SessionStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
                logger.exception("dispatch failed for session %s", session.session_id)
            finally:
                if self._result_callback is not None:
                    result = session.to_result()
                    try:
                        outcome = self._result_callback(result)
                        if asyncio.iscoroutine(outcome):
                            await outcome
                    except Exception:
                        logger.exception("result callback raised")

    async def _stage_init(self, session: Session) -> None:
        session.mark(SessionStatus.INIT)
        t0 = time.time()
        spec = session.spec
        config = SandboxConfig(
            image=spec.image,
            bundle=spec.bundle,
            platform=spec.platform,
            env=spec.env,
        )
        sandbox = await self._deployment.create(config)
        session.sandbox_id = str(sandbox.sandbox_id)
        session.runtime_url = sandbox.runtime_url
        session.stage_durations_ms[DispatchStage.INIT] = (time.time() - t0) * 1000

    async def _stage_ready(self, session: Session) -> None:
        session.mark(SessionStatus.READY)

    async def _stage_running(self, session: Session) -> None:
        assert session.runtime_url is not None  # set in INIT
        spec = session.spec
        session.mark(SessionStatus.RUNNING)
        t0 = time.time()
        async with AsyncExitStack() as stack:
            client = RuntimeClient(session.runtime_url)
            await stack.enter_async_context(client)
            if self._host_namespace_factory is not None:
                client.register_namespace(self._host_namespace_factory(session))
            value = await client.remote(
                _resolve_callable(spec.callable_ref),
                *spec.args,
                **spec.kwargs,
            )
            session.value = value
        session.stage_durations_ms[DispatchStage.RUNNING] = (time.time() - t0) * 1000

    async def _stage_postrun(self, session: Session) -> None:
        t0 = time.time()
        try:
            if session.sandbox_id is not None:
                await self._deployment.delete(SandboxId(session.sandbox_id))
        finally:
            session.stage_durations_ms[DispatchStage.POSTRUN] = (time.time() - t0) * 1000

    async def join(self) -> None:
        """Wait for every in-flight session to terminate."""
        if not self._tasks:
            return
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)


def _resolve_callable(ref: str) -> Any:
    """`module::qualname` -> callable. Mirrors `RuntimeClient.remote`'s shape."""
    if "::" not in ref:
        raise ValueError(f"callable_ref must be 'module::qualname'; got {ref!r}")
    module_path, _, qualname = ref.partition("::")
    import importlib

    module = importlib.import_module(module_path)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


__all__ = ["Dispatcher", "DispatchStage", "ResultCallback"]
