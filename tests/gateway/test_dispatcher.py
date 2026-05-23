"""End-to-end Dispatcher test using a fake Deployment + RuntimeClient.

Tests verify:

  * Sessions move INIT -> READY -> RUNNING -> POSTRUN -> SUCCEEDED.
  * `pause()` blocks new RUNNING transitions; `resume()` releases.
  * Failed dispatches end in FAILED with the exception text in
    `error`.
  * `_resolve_callable` resolves `module::qualname`.
  * `result_callback` fires exactly once per session.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from agentix.deployment.base import Sandbox, SandboxConfig, SandboxId, SandboxInfo
from agentix.gateway.dispatcher import Dispatcher, DispatchStage, _resolve_callable
from agentix.gateway.session import SessionResult, SessionSpec, SessionStatus

# ── fakes ─────────────────────────────────────────────────────────────────


class FakeDeployment:
    """Minimal `Deployment` for tests — fast, no real containers."""

    def __init__(self) -> None:
        self._sandboxes: dict[str, Sandbox] = {}

    async def create(self, config: SandboxConfig) -> Sandbox:
        sid = SandboxId(f"sandbox-{uuid4().hex[:6]}")
        sb = Sandbox(sandbox_id=sid, runtime_url=f"http://127.0.0.1:0/{sid}", status="running")
        self._sandboxes[sid] = sb
        return sb

    async def delete(self, sandbox_id: SandboxId) -> None:
        self._sandboxes.pop(sandbox_id, None)

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        sb = self._sandboxes[sandbox_id]
        return SandboxInfo(sandbox_id=sandbox_id, runtime_url=sb.runtime_url, status="running")


# Patch the RuntimeClient used by the dispatcher to avoid socket IO.
class _FakeClient:
    def __init__(self, url: str, *_, **__):
        self._url = url

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def register_namespace(self, _ns: Any) -> None:
        return None

    async def remote(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Execute the callable directly so the dispatcher exercises
        # the resolution path without an SIO round-trip.
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result


# ── targets resolvable by `module::qualname` ──────────────────────────────


def _add(a: int, b: int) -> int:
    return a + b


async def _slow(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "done"


def _boom(msg: str = "boom") -> None:
    raise RuntimeError(msg)


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentix.gateway.dispatcher as disp

    monkeypatch.setattr(disp, "RuntimeClient", _FakeClient)


# ── tests ─────────────────────────────────────────────────────────────────


def test_resolve_callable_requires_double_colon() -> None:
    with pytest.raises(ValueError):
        _resolve_callable("module.no_marker")


def test_resolve_callable_dotted_qualname() -> None:
    # Resolve a real nested attribute to prove dotted qualnames work.
    fn = _resolve_callable("collections::OrderedDict.update")
    assert callable(fn)


@pytest.mark.asyncio
async def test_dispatch_runs_through_all_stages(fake_runtime) -> None:
    dispatcher = Dispatcher(deployment=FakeDeployment(), concurrency=2)
    spec = SessionSpec(
        callable_ref=f"{__name__}::_add",
        image="x",
        bundle="y",
        args=(2, 3),
    )
    session = dispatcher.dispatch(spec)
    await dispatcher.join()

    assert session.status is SessionStatus.SUCCEEDED
    assert session.value == 5
    # stage durations captured
    assert DispatchStage.INIT in session.stage_durations_ms
    assert DispatchStage.RUNNING in session.stage_durations_ms
    assert DispatchStage.POSTRUN in session.stage_durations_ms


@pytest.mark.asyncio
async def test_pause_blocks_new_running(fake_runtime) -> None:
    dispatcher = Dispatcher(deployment=FakeDeployment(), concurrency=2)
    dispatcher.pause()
    spec = SessionSpec(callable_ref=f"{__name__}::_add", image="x", bundle="y", args=(1, 1))
    session = dispatcher.dispatch(spec)

    # While paused the session should hover at READY (or earlier),
    # never reach RUNNING. Give the loop a few ticks.
    for _ in range(20):
        await asyncio.sleep(0.01)
        if session.status is SessionStatus.READY:
            break
    assert session.status in (SessionStatus.READY, SessionStatus.INIT)
    assert session.value is None

    dispatcher.resume()
    await dispatcher.join()
    assert session.status is SessionStatus.SUCCEEDED
    assert session.value == 2


@pytest.mark.asyncio
async def test_failed_session_reports_error(fake_runtime) -> None:
    dispatcher = Dispatcher(deployment=FakeDeployment())
    spec = SessionSpec(
        callable_ref=f"{__name__}::_boom",
        image="x",
        bundle="y",
        kwargs={"msg": "kapow"},
    )
    session = dispatcher.dispatch(spec)
    await dispatcher.join()
    assert session.status is SessionStatus.FAILED
    assert "kapow" in (session.error or "")


@pytest.mark.asyncio
async def test_result_callback_fires_once(fake_runtime) -> None:
    seen: list[SessionResult] = []

    async def cb(result: SessionResult) -> None:
        seen.append(result)

    dispatcher = Dispatcher(deployment=FakeDeployment(), result_callback=cb)
    spec = SessionSpec(callable_ref=f"{__name__}::_add", image="x", bundle="y", args=(4, 5))
    dispatcher.dispatch(spec)
    await dispatcher.join()
    assert len(seen) == 1
    assert seen[0].value == 9
    assert seen[0].status is SessionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_session_metadata_carries_through(fake_runtime) -> None:
    dispatcher = Dispatcher(deployment=FakeDeployment())
    spec = SessionSpec(
        callable_ref=f"{__name__}::_add",
        image="x",
        bundle="y",
        args=(0, 0),
        metadata={"instance_id": "django-1"},
    )
    session = dispatcher.dispatch(spec)
    await dispatcher.join()
    result = session.to_result()
    assert result.metadata == {"instance_id": "django-1"}
