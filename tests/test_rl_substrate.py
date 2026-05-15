"""Tests for the RL substrate: trace pipeline, LLM proxy, RolloutPool.

The first wave of RL-loop primitives. Uses the same `live_server` fixture
as the streaming/bidi tests so trace events flow over a real Socket.IO
connection.
"""

from __future__ import annotations

import asyncio
import json
import textwrap

import httpx
import pytest

from agentix import RuntimeClient
from agentix.runtime.models import RemoteRequest, TraceEvent

pytestmark = pytest.mark.asyncio


# ── trace pipeline ──────────────────────────────────────────────


_TRACE_INIT = textwrap.dedent("""
    from agentix import trace
    def step(label: str) -> int:
        # The dispatcher pins call_id + source via contextvar before invoking
        # this impl, so `trace.emit` automatically inherits the correlation.
        # In stubs we just declare the contract; the body raises.
        raise NotImplementedError
""")

_TRACE_IMPL = textwrap.dedent("""
    from agentix import trace
    def step(label):
        trace.emit("tool_call", {"tool": "echo", "arg": label})
        trace.emit("reward", {"value": 0.5})
        return 42
""")

_TRACE_REGISTER = textwrap.dedent("""
    from agentix.dispatch import Dispatcher
    from . import step
    from ._impl import step as _step
    def register():
        d = Dispatcher(); d.bind(step, _step); return d
""")


async def test_trace_emit_received_by_subscriber(
    runtime_module, mount_package, live_server,
):
    """A closure emits two trace events; subscriber receives them with the
    dispatcher-pinned call_id + source filled in."""
    mount_package(
        "tracer",
        package="agentix_closures.tracer",
        init_src=_TRACE_INIT, impl_src=_TRACE_IMPL, register_src=_TRACE_REGISTER,
    )
    base_url = await live_server()

    async with RuntimeClient(base_url) as c:
        received: list[TraceEvent] = []

        async def _collect():
            async for ev in c.traces():
                received.append(ev)
                if len(received) >= 2:
                    return

        collector = asyncio.create_task(_collect())
        # Let the subscription land server-side.
        await asyncio.sleep(0.2)
        # Drive a unary call with a call_id so the trace inherits it.
        body = RemoteRequest(
            package="agentix_closures.tracer", method="step",
            kwargs={"label": "hi"}, call_id="rollout-7",
        ).model_dump()
        async with httpx.AsyncClient(base_url=base_url) as http:
            r = await http.post("/_remote", json=body)
            assert r.status_code == 200
            assert r.json() == {"ok": True, "value": 42, "error": None}

        await asyncio.wait_for(collector, timeout=3)

    kinds = [e.kind for e in received]
    assert kinds == ["tool_call", "reward"]
    assert all(e.call_id == "rollout-7" for e in received)
    assert all(e.source == "agentix_closures.tracer" for e in received)
    assert received[0].payload == {"tool": "echo", "arg": "hi"}
    assert received[1].payload == {"value": 0.5}


async def test_trace_filter_by_kind(runtime_module, mount_package, live_server):
    """`c.traces(kind=...)` only yields matching events."""
    mount_package(
        "tracer2",
        package="agentix_closures.tracer2",
        init_src=_TRACE_INIT, impl_src=_TRACE_IMPL, register_src=_TRACE_REGISTER,
    )
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        received: list[TraceEvent] = []

        async def _collect():
            async for ev in c.traces(kind="reward"):
                received.append(ev)
                return

        collector = asyncio.create_task(_collect())
        await asyncio.sleep(0.2)
        body = RemoteRequest(
            package="agentix_closures.tracer2", method="step",
            kwargs={"label": "x"}, call_id="r-1",
        ).model_dump()
        async with httpx.AsyncClient(base_url=base_url) as http:
            await http.post("/_remote", json=body)
        await asyncio.wait_for(collector, timeout=3)

    assert len(received) == 1
    assert received[0].kind == "reward"


# ── LLM proxy ───────────────────────────────────────────────────


async def test_llm_proxy_forwards_and_traces(runtime_module, live_server, monkeypatch):
    """Proxy reaches the upstream, returns the body, and emits llm_request /
    llm_response events. Stand up a tiny fake `api.anthropic.com` via httpx
    MockTransport in the same process; switch the proxy's upstream table to
    point at the in-process mock URL.
    """
    base_url = await live_server()

    # Replace the proxy's upstream map to point at an httpx mock app.
    # We can't easily inject a MockTransport into the proxy's httpx.AsyncClient
    # (it builds one per request). Instead, point the upstream at a tiny
    # fake server running in the same process on another port.
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from agentix.runtime.server import builtins as _builtins

    async def _fake_messages(request):
        body = await request.body()
        payload = json.loads(body) if body else {}
        return JSONResponse({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "echo: " + payload.get("prompt", "")}],
            "model": payload.get("model", "claude-test"),
            "usage": {"input_tokens": 3, "output_tokens": 4},
        })

    fake = Starlette(routes=[Route("/v1/messages", _fake_messages, methods=["POST"])])
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        fake_port = s.getsockname()[1]
    fake_cfg = uvicorn.Config(fake, host="127.0.0.1", port=fake_port, log_level="error")
    fake_srv = uvicorn.Server(fake_cfg)
    fake_task = asyncio.create_task(fake_srv.serve())
    fake_url = f"http://127.0.0.1:{fake_port}"
    async with httpx.AsyncClient() as probe:
        for _ in range(40):
            try:
                r = await probe.get(f"{fake_url}/v1/messages")
                if r.status_code < 500:
                    break
            except (httpx.ConnectError, httpx.ReadError):
                await asyncio.sleep(0.05)

    monkeypatch.setitem(_builtins._LLM_UPSTREAMS, "anthropic", fake_url)

    try:
        async with RuntimeClient(base_url) as c:
            received: list[TraceEvent] = []

            async def _collect():
                async for ev in c.traces():
                    received.append(ev)
                    if len(received) >= 2:
                        return

            collector = asyncio.create_task(_collect())
            await asyncio.sleep(0.2)

            # Caller (e.g. claude-code) hits the proxy URL just like it would
            # hit api.anthropic.com directly.
            async with httpx.AsyncClient(base_url=base_url) as http:
                r = await http.post(
                    "/_llm/anthropic/v1/messages",
                    json={"prompt": "hello", "model": "claude-test"},
                )
                assert r.status_code == 200
                body = r.json()
                assert body["content"][0]["text"] == "echo: hello"

            await asyncio.wait_for(collector, timeout=3)
    finally:
        fake_srv.should_exit = True
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fake_task), timeout=0.01)
        try:
            await asyncio.wait_for(fake_task, timeout=5)
        except TimeoutError:
            fake_task.cancel()

    kinds = [e.kind for e in received]
    assert kinds == ["llm_request", "llm_response"]
    assert received[0].payload["provider"] == "anthropic"
    assert received[0].payload["path"] == "/v1/messages"
    assert received[1].payload["status"] == 200


async def test_llm_proxy_rejects_unknown_provider(runtime_module, live_server):
    base_url = await live_server()
    async with httpx.AsyncClient(base_url=base_url) as http:
        r = await http.get("/_llm/cohere/whatever")
    assert r.status_code == 404
    assert "unknown LLM provider" in r.json()["detail"]


# ── RolloutPool ─────────────────────────────────────────────────


async def test_rollout_pool_maps_and_returns_results(
    runtime_module, mount_echo, live_server,
):
    """RolloutPool fans out N tasks across parallelism slots and yields results."""
    # We can't (cheaply) spin up real DockerDeployment in unit tests, so
    # test the pool against a tiny in-memory deployment that hands back the
    # already-running live_server URL for every "sandbox".
    mount_echo()
    base_url = await live_server()
    from agentix_closures.echo import echo

    class _StubSandbox:
        def __init__(self, sid: str, url: str):
            self.sandbox_id = sid
            self.runtime_url = url
            self.status = "running"

    class _StubDeployment:
        def __init__(self, url: str):
            self.url = url
            self.created: list[str] = []
            self.deleted: list[str] = []

        async def create(self, _config):
            sid = f"stub-{len(self.created)}"
            self.created.append(sid)
            return _StubSandbox(sid, self.url)

        async def delete(self, sid: str):
            self.deleted.append(sid)

        async def get(self, sid):
            return _StubSandbox(sid, self.url)

    from agentix import SandboxConfig
    from agentix.rollout import RolloutPool

    deployment = _StubDeployment(base_url)
    config = SandboxConfig(image="x", runtime="y")

    async def _run_one(client, task: str):
        r = await client.remote(echo, msg=task)
        return r.msg

    async with RolloutPool(deployment, config, parallelism=3) as pool:
        assert len(deployment.created) == 3
        tasks = ["a", "b", "c", "d", "e"]
        results = []
        async for task, result in pool.map(_run_one, tasks):
            assert not isinstance(result, BaseException)
            results.append((task, result))

    assert sorted(results) == [
        ("a", "echo:a"), ("b", "echo:b"), ("c", "echo:c"),
        ("d", "echo:d"), ("e", "echo:e"),
    ]
    assert len(deployment.deleted) == 3


async def test_rollout_pool_surfaces_per_task_errors(runtime_module, mount_package, live_server):
    """One bad task doesn't tank the pool — its exception is yielded as the result."""
    init_src = textwrap.dedent("""
        def explode_on(label: str) -> str:
            raise NotImplementedError
    """)
    impl_src = textwrap.dedent("""
        def explode_on(label):
            if label == "bad":
                raise ValueError("nope")
            return f"ok:{label}"
    """)
    register_src = textwrap.dedent("""
        from agentix.dispatch import Dispatcher
        from . import explode_on
        from ._impl import explode_on as _e
        def register():
            d = Dispatcher(); d.bind(explode_on, _e); return d
    """)
    mount_package(
        "splody",
        package="agentix_closures.splody",
        init_src=init_src, impl_src=impl_src, register_src=register_src,
    )
    base_url = await live_server()
    from agentix_closures.splody import explode_on

    class _StubSandbox:
        def __init__(self, sid, url):
            self.sandbox_id = sid
            self.runtime_url = url
            self.status = "running"

    class _StubDeployment:
        async def create(self, _config):
            return _StubSandbox("s", base_url)
        async def delete(self, _sid): pass
        async def get(self, sid): return _StubSandbox(sid, base_url)

    from agentix import RemoteCallError, SandboxConfig
    from agentix.rollout import RolloutPool

    async def _run_one(client, task):
        return await client.remote(explode_on, label=task)

    async with RolloutPool(
        _StubDeployment(), SandboxConfig(image="x", runtime="y"), parallelism=2,
    ) as pool:
        ok = []
        errors = []
        async for task, result in pool.map(_run_one, ["a", "bad", "c"]):
            if isinstance(result, BaseException):
                errors.append((task, result))
            else:
                ok.append((task, result))

    assert sorted(ok) == [("a", "ok:a"), ("c", "ok:c")]
    assert len(errors) == 1
    assert errors[0][0] == "bad"
    assert isinstance(errors[0][1], RemoteCallError)
    assert errors[0][1].error.type == "ValueError"
