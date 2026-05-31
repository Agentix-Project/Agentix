"""`bridged()` lifecycle + session-grouped trajectory.

`bridged` is the sandbox-side entry the host invokes via
`sandbox.remote(bridged, agent_fn, ...)`: it brings the proxy up around
the agent, points env at it, dispatches sync agents to a thread, and
tears down on exit — all without the agent touching env or proxy
lifecycle. Here we monkeypatch start/stop so no real uvicorn is needed.
"""

from __future__ import annotations

import os
import threading

import agentix.bridge.proxy as proxy_mod
import pytest
from agentix.bridge import BridgeConfig, CompletionRecord, InMemoryStore, bridged
from agentix.bridge.detection import ApiFamily


@pytest.fixture
def fake_proxy(monkeypatch) -> dict:
    state: dict = {"started_with": None, "stopped": False}
    handle = proxy_mod.ProxyHandle(
        proxy_id="test",
        url="http://127.0.0.1:9999",
        port=9999,
        anthropic_base_url="http://127.0.0.1:9999",
        openai_base_url="http://127.0.0.1:9999/v1",
    )

    async def fake_start(**kwargs):
        state["started_with"] = kwargs
        return handle

    async def fake_stop(h):
        state["stopped"] = True

    monkeypatch.setattr(proxy_mod, "start_proxy", fake_start)
    monkeypatch.setattr(proxy_mod, "stop_proxy", fake_stop)
    # Keep the test process's env clean.
    for key in ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return state


@pytest.mark.asyncio
async def test_bridged_runs_sync_agent_in_thread_and_sets_env(fake_proxy) -> None:
    main = threading.get_ident()

    def agent(task: str) -> dict:
        # Sync agents must run off the event-loop thread so the proxy's
        # uvicorn server stays free to serve their HTTP calls.
        return {
            "task": task,
            "off_loop_thread": threading.get_ident() != main,
            "base_url": os.environ.get("ANTHROPIC_BASE_URL"),
            "key": os.environ.get("OPENAI_API_KEY"),
        }

    result = await bridged(agent, task="hi", _bridge=BridgeConfig(session_id="s1"))

    assert result["task"] == "hi"
    assert result["off_loop_thread"] is True
    assert result["base_url"] == "http://127.0.0.1:9999"
    assert result["key"] == "sk-abridge"  # placeholder, never the real key
    assert fake_proxy["started_with"]["session_id"] == "s1"
    assert fake_proxy["stopped"] is True


@pytest.mark.asyncio
async def test_bridged_awaits_async_agent_and_tears_down_on_error(fake_proxy) -> None:
    async def boom() -> None:
        raise ValueError("agent failed")

    with pytest.raises(ValueError, match="agent failed"):
        await bridged(boom)

    # Teardown happens even when the agent raises.
    assert fake_proxy["stopped"] is True


@pytest.mark.asyncio
async def test_bridged_generates_session_id_when_unset(fake_proxy) -> None:
    await bridged(lambda: None)
    assert fake_proxy["started_with"]["session_id"]  # a generated id, not None


def test_store_trajectory_groups_by_session() -> None:
    store = InMemoryStore()

    def rec(rid: str, sid: str | None) -> CompletionRecord:
        return CompletionRecord(
            request_id=rid,
            family=ApiFamily.OPENAI_CHAT_COMPLETIONS,
            started_at=0.0,
            ended_at=1.0,
            request_path="/v1/chat/completions",
            request_body={},
            upstream_body={},
            response_body={},
            session_id=sid,
        )

    store.add(rec("a", "s1"))
    store.add(rec("b", "s2"))
    store.add(rec("c", "s1"))
    store.add(rec("d", None))

    traj = store.trajectory("s1")
    assert [r.request_id for r in traj] == ["a", "c"]  # order preserved
    assert store.sessions() == ["s1", "s2"]  # first-appearance order, None skipped
