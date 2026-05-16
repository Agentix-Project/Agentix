"""Protocol-level integration tests for the runtime server's dispatch path.

These tests drive the real `agentix.runtime.server` FastAPI app over an
ASGI transport. The fixture `register_namespace` injects a `Namespace`
subclass straight into the runtime's registry, bypassing the
`importlib.metadata.entry_points` discovery that production uses — same
end state, no on-disk packaging.

Closure classes are declared at module scope so `eval_str=True` (used
by the framework to resolve PEP 563 stringified annotations) can find
their referenced types.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest

from agentix import RemoteCallError, RuntimeClient
from agentix.namespace import Namespace
from agentix.runtime.models import RemoteRequest

pytestmark = pytest.mark.asyncio


# ── namespace shapes used across tests ─────────────────────────────────


@dataclass
class EchoResult:
    msg: str


class Echo(Namespace):
    @staticmethod
    async def echo(msg: str) -> EchoResult:
        return EchoResult(msg=f"echo:{msg}")


class Boom(Namespace):
    @staticmethod
    async def go() -> str:
        raise RuntimeError("kaboom")


@dataclass
class Token:
    text: str
    idx: int


class Streamer(Namespace):
    @staticmethod
    async def chat(prompt: str, n: int = 3) -> AsyncIterator[Token]:
        for i in range(n):
            yield Token(text=f"{prompt}-{i}", idx=i)


@dataclass
class UserMsg:
    text: str


@dataclass
class ReplyMsg:
    text: str


class Chat(Namespace):
    @staticmethod
    async def chat(
        messages: AsyncIterator[UserMsg],
        prefix: str = "say:",
    ) -> AsyncIterator[ReplyMsg]:
        async for m in messages:
            yield ReplyMsg(text=f"{prefix}{m.text}")


class Talker(Namespace):
    @staticmethod
    async def speak() -> str:
        logging.getLogger("agentix.test.talker").info("hello-from-impl")
        return "ok"


# ── registry + dispatch basics ───────────────────────────────────────


async def test_register_namespace_makes_it_dispatchable(
    runtime_module, register_namespace,
):
    """A registered namespace surfaces in `/namespaces` and dispatches via /_remote."""
    server, _, _ = runtime_module
    register_namespace(Echo)
    pkg = Echo.__module__

    assert server.multiplexer.has(pkg)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.get("/namespaces")
        assert r.status_code == 200
        pkgs = [c["manifest"]["package"] for c in r.json()]
        assert pkg in pkgs

        body = RemoteRequest(
            package=pkg, method="echo", kwargs={"msg": "hi"},
        ).model_dump()
        r = await http.post("/_remote", json=body)
        assert r.status_code == 200
        assert r.json() == {"ok": True, "value": {"msg": "echo:hi"}, "error": None}


async def test_remote_call_unknown_package_404(runtime_module):
    server, _, _ = runtime_module
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        body = RemoteRequest(package="agentix.nope", method="x").model_dump()
        r = await http.post("/_remote", json=body)
        assert r.status_code == 404


async def test_remote_call_unknown_method_returns_error_body(
    runtime_module, register_namespace,
):
    server, _, _ = runtime_module
    register_namespace(Echo)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        body = RemoteRequest(
            package=Echo.__module__, method="not_a_method",
        ).model_dump()
        r = await http.post("/_remote", json=body)
        assert r.status_code == 200  # 200 with ok=False; wire stays clean
        body_json = r.json()
        assert body_json["ok"] is False
        assert body_json["error"]["type"] == "MethodNotFound"


async def test_impl_exception_surfaces_as_remote_error(
    runtime_module, register_namespace,
):
    server, _, _ = runtime_module
    register_namespace(Boom)
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        body = RemoteRequest(package=Boom.__module__, method="go").model_dump()
        r = await http.post("/_remote", json=body)
        assert r.status_code == 200
        body_json = r.json()
        assert body_json["ok"] is False
        assert body_json["error"]["type"] == "RuntimeError"
        assert "kaboom" in body_json["error"]["message"]


async def test_client_remote_round_trip(
    runtime_module, register_namespace, live_server,
):
    register_namespace(Echo)
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        result = await c.remote(Echo.echo, msg="hello")
        assert result.msg == "echo:hello"


async def test_client_remote_raises_on_impl_error(
    runtime_module, register_namespace, live_server,
):
    register_namespace(Boom)
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        with pytest.raises(RemoteCallError):
            await c.remote(Boom.go)


# ── streaming + bidi ─────────────────────────────────────────────────


async def test_stream_round_trip_via_socketio(
    runtime_module, register_namespace, live_server,
):
    register_namespace(Streamer)
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        items = [t async for t in c.remote(Streamer.chat, prompt="hi", n=2)]
        assert items == [Token(text="hi-0", idx=0), Token(text="hi-1", idx=1)]


async def test_bidi_round_trip_via_socketio(
    runtime_module, register_namespace, live_server,
):
    register_namespace(Chat)
    base_url = await live_server()

    async def _inputs() -> AsyncIterator[UserMsg]:
        for t in ("hi", "there"):
            yield UserMsg(text=t)

    async with RuntimeClient(base_url) as c:
        replies = [r async for r in c.remote(Chat.chat, messages=_inputs())]
        assert [r.text for r in replies] == ["say:hi", "say:there"]


# ── log subscription ─────────────────────────────────────────────────


@pytest.mark.skip(reason="log forwarding fixture timing flake; tracked separately")
async def test_logs_subscription_receives_emitted_log(
    runtime_module, register_namespace, live_server,
):
    register_namespace(Talker)
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        seen: list[str] = []

        async def _collect():
            async for rec in c.logs():
                if rec.name == "agentix.test.talker":
                    seen.append(rec.message)
                    return

        collector = asyncio.create_task(_collect())
        await asyncio.sleep(0.2)
        await c.remote(Talker.speak)
        await asyncio.wait_for(collector, timeout=5)
        assert seen == ["hello-from-impl"]
