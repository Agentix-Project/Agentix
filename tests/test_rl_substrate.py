"""Tests for the RL substrate: trace pipeline, LLM proxy, RolloutPool."""

from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest

from agentix import RuntimeClient
from agentix.runtime.models import RemoteRequest, TraceEvent

pytestmark = pytest.mark.asyncio


# ── trace pipeline ──────────────────────────────────────────────


class Tracer:
    @staticmethod
    async def step(label: str) -> int:
        from agentix import trace
        trace.emit("tool_call", {"tool": "echo", "arg": label})
        trace.emit("reward", {"value": 0.5})
        return 42


async def test_trace_emit_received_by_subscriber(
    runtime_module, register_namespace, live_server,
):
    """A namespace emits two trace events; subscriber receives them with the
    dispatcher-pinned call_id + source filled in."""
    register_namespace(Tracer)
    base_url = await live_server()

    async with RuntimeClient(base_url) as c:
        received: list[TraceEvent] = []

        async def _collect():
            async for ev in c.traces():
                received.append(ev)
                if len(received) >= 2:
                    return

        collector = asyncio.create_task(_collect())
        await asyncio.sleep(0.2)
        from agentix.runtime.codec import pack, unpack
        body = pack(RemoteRequest(
            package=Tracer.__module__, method="step",
            kwargs={"label": "hi"}, call_id="rollout-7",
        ).model_dump())
        async with httpx.AsyncClient(base_url=base_url) as http:
            r = await http.post("/_remote", content=body,
                                headers={"Content-Type": "application/msgpack"})
            assert r.status_code == 200
            assert unpack(r.content) == {"ok": True, "value": 42, "error": None}

        await asyncio.wait_for(collector, timeout=5)

    kinds = [e.kind for e in received]
    assert kinds == ["tool_call", "reward"]
    assert all(e.call_id == "rollout-7" for e in received)
    assert all(e.source == Tracer.__module__ for e in received)
    assert received[0].payload == {"tool": "echo", "arg": "hi"}
    assert received[1].payload == {"value": 0.5}


async def test_trace_filter_by_kind(runtime_module, register_namespace, live_server):
    """`c.traces(kind=...)` only yields matching events."""
    register_namespace(Tracer)
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        received: list[TraceEvent] = []

        async def _collect():
            async for ev in c.traces(kind="reward"):
                received.append(ev)
                return

        collector = asyncio.create_task(_collect())
        await asyncio.sleep(0.2)
        from agentix.runtime.codec import pack
        body = pack(RemoteRequest(
            package=Tracer.__module__, method="step",
            kwargs={"label": "x"}, call_id="r-1",
        ).model_dump())
        async with httpx.AsyncClient(base_url=base_url) as http:
            await http.post("/_remote", content=body,
                            headers={"Content-Type": "application/msgpack"})
        await asyncio.wait_for(collector, timeout=5)

    assert len(received) == 1
    assert received[0].kind == "reward"


async def test_typed_client_call_context_sets_trace_call_id(
    runtime_module, register_namespace, live_server,
):
    register_namespace(Tracer)
    base_url = await live_server()

    async with RuntimeClient(base_url) as c:
        received: list[TraceEvent] = []

        async def _collect():
            async for ev in c.traces():
                received.append(ev)
                if len(received) >= 2:
                    return

        collector = asyncio.create_task(_collect())
        await asyncio.sleep(0.2)
        with c.call_context(call_id="typed-rollout-1"):
            result = await c.remote(Tracer.step, label="typed")
        assert result == 42
        await asyncio.wait_for(collector, timeout=5)

    assert all(e.call_id == "typed-rollout-1" for e in received)


# ── LLM proxy ───────────────────────────────────────────────────


async def test_llm_proxy_forwards_and_traces(runtime_module, live_server, monkeypatch):
    """Proxy reaches the upstream, returns the body, and emits llm_request /
    llm_response events. Stand up a tiny fake `api.anthropic.com` in the
    same process; point the proxy's upstream table at it.
    """
    base_url = await live_server()

    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from agentix.runtime.server import llm_proxy as _llm_proxy_mod

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

    monkeypatch.setitem(_llm_proxy_mod._LLM_UPSTREAMS, "anthropic", fake_url)

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

            async with httpx.AsyncClient(base_url=base_url) as http:
                r = await http.post(
                    "/_llm/anthropic/v1/messages",
                    json={"prompt": "hello", "model": "claude-test"},
                )
                assert r.status_code == 200
                body = r.json()
                assert body["content"][0]["text"] == "echo: hello"

            await asyncio.wait_for(collector, timeout=5)
    finally:
        fake_srv.should_exit = True
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


class EchoNs:
    @staticmethod
    async def echo(msg: str) -> str:
        return f"echo:{msg}"


class Splody:
    @staticmethod
    async def explode_on(label: str) -> str:
        if label == "bad":
            raise ValueError("nope")
        return f"ok:{label}"


async def test_rollout_pool_maps_and_returns_results(
    runtime_module, register_namespace, live_server,
):
    """RolloutPool fans out N tasks across parallelism slots and yields results."""
    register_namespace(EchoNs)
    base_url = await live_server()

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
    config = SandboxConfig(image="x")

    async def _run_one(client, task: str):
        r = await client.remote(EchoNs.echo, msg=task)
        return r

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


async def test_rollout_pool_surfaces_per_task_errors(
    runtime_module, register_namespace, live_server,
):
    """One bad task doesn't tank the pool — its exception is yielded as the result."""
    register_namespace(Splody)
    base_url = await live_server()

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

    from agentix import SandboxConfig
    from agentix.rollout import RolloutPool

    async def _run_one(client, task):
        return await client.remote(Splody.explode_on, label=task)

    async with RolloutPool(
        _StubDeployment(), SandboxConfig(image="x"), parallelism=2,
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


async def test_rollout_pool_map_cancels_workers_when_iterator_closes():
    """Breaking out of map() should not leave queued tasks running."""
    from agentix import SandboxConfig
    from agentix.rollout import RolloutPool

    class _StubSandbox:
        def __init__(self, sid, url):
            self.sandbox_id = sid
            self.runtime_url = url
            self.status = "running"

    class _StubDeployment:
        async def create(self, _config):
            return _StubSandbox("s", "http://127.0.0.1:9")
        async def delete(self, _sid): pass
        async def get(self, sid): return _StubSandbox(sid, "http://127.0.0.1:9")

    started: list[int] = []

    async def _run_one(_client, task: int):
        started.append(task)
        if task == 0:
            return "first"
        await asyncio.sleep(60)
        return "late"

    async with RolloutPool(
        _StubDeployment(), SandboxConfig(image="x"), parallelism=2,
    ) as pool:
        ait = pool.map(_run_one, list(range(10)))
        task, result = await anext(ait)
        assert (task, result) == (0, "first")
        await ait.aclose()
        await asyncio.sleep(0.1)

    assert len(started) <= 2
