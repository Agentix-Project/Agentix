"""Protocol-level integration tests for the runtime server's dispatch path.

These tests drive the real `agentix.runtime.server` FastAPI app over an
ASGI transport. There is no subprocess, no UDS, no reverse proxy — the
runtime imports each mounted closure's Python package in-process and
serves `POST /_remote` by calling the bound impl directly.
"""

from __future__ import annotations

import asyncio
import textwrap

import httpx
import pytest

from agentix import RemoteCallError, RuntimeClient
from agentix.runtime.models import RemoteRequest

pytestmark = pytest.mark.asyncio


async def test_auto_load_registers_package(runtime_module, mount_echo):
    """A valid mount auto-loads under `manifest.package`."""
    server, _root, _up = runtime_module
    mount_echo("echo")
    await server._auto_load()
    assert "agentix_closures.echo" in server.registry
    assert server.registry.packages() == ["agentix_closures.echo"]


async def test_auto_load_skips_runtime_dir(runtime_module, mount_root_setup, mount_echo):
    """A mount named 'runtime' is reserved and skipped."""
    server, _root, _up = runtime_module
    mount_echo("runtime")  # would-be closure under the reserved name
    await server._auto_load()
    assert "agentix_closures.echo" not in server.registry


async def test_auto_load_skips_without_manifest(runtime_module, mount_root_setup):
    """A /mnt/<dir> with no entry/manifest.json is ignored (not a closure)."""
    server, root, _ = runtime_module
    bogus = root / "bogus"
    (bogus / "entry").mkdir(parents=True)
    await server._auto_load()
    assert server.registry.packages() == []


async def test_auto_load_skips_wrong_abi(runtime_module, mount_package):
    server, _, _ = runtime_module
    mount_package(
        "future",
        package="agentix_closures.future",
        init_src="def x(): ...",
        impl_src="def x(): return 1",
        register_src=textwrap.dedent("""
            from agentix.dispatch import Dispatcher
            from . import x
            from ._impl import x as _x
            def register():
                d = Dispatcher()
                d.bind(x, _x)
                return d
        """),
        abi=999,
    )
    await server._auto_load()
    assert server.registry.packages() == []


async def test_two_closures_coexist(runtime_module, mount_echo, mount_package):
    server, _, _ = runtime_module
    mount_echo("c0")
    mount_package(
        "c1",
        package="agentix_closures.greet",
        init_src=textwrap.dedent("""
            from dataclasses import dataclass
            @dataclass
            class HiResult: text: str
            def hi(name: str) -> HiResult: ...
        """),
        impl_src=textwrap.dedent("""
            from . import HiResult
            def hi(name): return HiResult(text=f"hi {name}")
        """),
        register_src=textwrap.dedent("""
            from agentix.dispatch import Dispatcher
            from . import hi
            from ._impl import hi as _hi
            def register():
                d = Dispatcher(); d.bind(hi, _hi); return d
        """),
    )
    await server._auto_load()
    assert set(server.registry.packages()) == {
        "agentix_closures.echo",
        "agentix_closures.greet",
    }


async def test_duplicate_package_collides(runtime_module, mount_echo):
    """Two mounts shipping the same package: second is skipped."""
    server, _, _ = runtime_module
    mount_echo("c0", package="agentix_closures.echo")
    mount_echo("c1", package="agentix_closures.echo")
    await server._auto_load()
    assert server.registry.packages() == ["agentix_closures.echo"]


# ── wire ─────────────────────────────────────────────────────────


async def test_remote_dispatches_to_impl(runtime_module, mount_echo):
    server, _, _ = runtime_module
    mount_echo()
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        body = RemoteRequest(
            package="agentix_closures.echo", method="echo", kwargs={"msg": "hi"}
        ).model_dump()
        r = await http.post("/_remote", json=body)
        assert r.status_code == 200
        resp = r.json()
        assert resp["ok"] is True
        assert resp["value"] == {"msg": "echo:hi"}


async def test_remote_method_not_found(runtime_module, mount_echo):
    server, _, _ = runtime_module
    mount_echo()
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post(
            "/_remote",
            json={"package": "agentix_closures.echo", "method": "bogus"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["error"]["type"] == "MethodNotFound"


async def test_remote_validation_error(runtime_module, mount_echo):
    server, _, _ = runtime_module
    mount_echo()
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        # `msg` is typed `str`; pass an int that pydantic refuses to coerce in strict.
        r = await http.post(
            "/_remote",
            json={
                "package": "agentix_closures.echo",
                "method": "echo",
                "kwargs": {"msg": {"not": "a string"}},
            },
        )
        body = r.json()
        assert body["ok"] is False
        assert body["error"]["type"] == "ValidationError"


async def test_remote_impl_raises(runtime_module, mount_package):
    server, _, _ = runtime_module
    mount_package(
        "boom",
        package="agentix_closures.boom",
        init_src="def explode() -> int: ...",
        impl_src="def explode(): raise ValueError('kaboom')",
        register_src=textwrap.dedent("""
            from agentix.dispatch import Dispatcher
            from . import explode
            from ._impl import explode as _e
            def register():
                d = Dispatcher(); d.bind(explode, _e); return d
        """),
    )
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post(
            "/_remote",
            json={"package": "agentix_closures.boom", "method": "explode"},
        )
        body = r.json()
        assert body["ok"] is False
        assert body["error"]["type"] == "ValueError"
        assert "kaboom" in body["error"]["message"]


async def test_remote_unknown_package(runtime_module):
    server, _, _ = runtime_module
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post(
            "/_remote",
            json={"package": "agentix_closures.nope", "method": "x"},
        )
        assert r.status_code == 404


async def test_closures_inventory(runtime_module, mount_echo):
    server, _, _ = runtime_module
    mount_echo()
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.get("/closures")
        assert r.status_code == 200
        items = r.json()
        assert len(items) == 1
        assert items[0]["manifest"]["package"] == "agentix_closures.echo"


# ── typed client ─────────────────────────────────────────────────


async def test_runtime_client_remote_typed(runtime_module, mount_echo):
    """`RuntimeClient.remote(fn, ...)` returns the stub's return type."""
    server, _, _ = runtime_module
    mount_echo()
    await server._auto_load()

    # _auto_load adds <mount>/entry/python to sys.path; the stub is now importable.
    from agentix_closures.echo import EchoResult, echo

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        client = RuntimeClient.__new__(RuntimeClient)
        client._client = http  # use the ASGI transport directly
        result = await client.remote(echo, msg="world")
    assert isinstance(result, EchoResult)
    assert result.msg == "echo:world"


async def test_runtime_client_propagates_remote_error(runtime_module, mount_package):
    server, _, _ = runtime_module
    mount_package(
        "boom",
        package="agentix_closures.boom2",
        init_src="def explode() -> int: ...",
        impl_src="def explode(): raise RuntimeError('x')",
        register_src=textwrap.dedent("""
            from agentix.dispatch import Dispatcher
            from . import explode
            from ._impl import explode as _e
            def register():
                d = Dispatcher(); d.bind(explode, _e); return d
        """),
    )
    await server._auto_load()

    from agentix_closures.boom2 import explode

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        client = RuntimeClient.__new__(RuntimeClient)
        client._client = http
        with pytest.raises(RemoteCallError) as ei:
            await client.remote(explode)
    assert ei.value.error.type == "RuntimeError"


# ── lazy load ────────────────────────────────────────────────────


async def test_register_is_pending_until_first_call(runtime_module, mount_echo):
    """`_auto_load` only marks the closure pending — no Dispatcher built."""
    server, _, _ = runtime_module
    mount_echo()
    await server._auto_load()
    # Known package, but dispatcher not yet built.
    assert "agentix_closures.echo" in server.registry
    assert server.registry.packages() == ["agentix_closures.echo"]
    assert server.registry.loaded_packages() == []


async def test_lazy_load_on_first_remote(runtime_module, mount_echo):
    """First `/_remote` call materialises the Dispatcher; second call reuses it."""
    server, _, _ = runtime_module
    mount_echo()
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        body = {"package": "agentix_closures.echo", "method": "echo", "kwargs": {"msg": "x"}}
        r = await http.post("/_remote", json=body)
        assert r.status_code == 200
        assert server.registry.loaded_packages() == ["agentix_closures.echo"]
        # Same dispatcher instance reused on subsequent calls.
        d1 = await server.registry.get_or_load("agentix_closures.echo")
        r = await http.post("/_remote", json=body)
        assert r.status_code == 200
        d2 = await server.registry.get_or_load("agentix_closures.echo")
        assert d1 is d2


async def test_lazy_load_failure_cached(runtime_module, mount_package):
    """A closure whose `_register.register()` raises caches the error and
    re-raises on every subsequent call (no retry storm)."""
    server, _, _ = runtime_module
    mount_package(
        "broken",
        package="agentix_closures.broken_reg",
        init_src="def x(): ...",
        impl_src="def x(): return 1",
        register_src=textwrap.dedent("""
            def register():
                raise RuntimeError("register-side boom")
        """),
    )
    await server._auto_load()

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        body = {"package": "agentix_closures.broken_reg", "method": "x"}
        r1 = await http.post("/_remote", json=body)
        r2 = await http.post("/_remote", json=body)
    assert r1.status_code == 500
    assert "register-side boom" in r1.json()["detail"]
    # Second call hits the cached error, same response (no second import attempt).
    assert r2.status_code == 500
    assert r2.json()["detail"] == r1.json()["detail"]


async def test_concurrent_first_calls_serialise(runtime_module, mount_package):
    """Two simultaneous first-time `/_remote` calls share a single import."""
    server, _, _ = runtime_module
    # The register module records how many times register() is called.
    impl_src = textwrap.dedent("""
        import time
        from . import x
        async def x():
            return 42
    """)
    register_src = textwrap.dedent("""
        from agentix.dispatch import Dispatcher
        from . import x
        from ._impl import x as _x
        _call_count = 0
        def register():
            global _call_count
            _call_count += 1
            d = Dispatcher(); d.bind(x, _x); return d
    """)
    mount_package(
        "race",
        package="agentix_closures.race_pkg",
        init_src="async def x() -> int: ...",
        impl_src=impl_src,
        register_src=register_src,
    )
    await server._auto_load()

    import asyncio as _asyncio
    results = await _asyncio.gather(
        server.registry.get_or_load("agentix_closures.race_pkg"),
        server.registry.get_or_load("agentix_closures.race_pkg"),
        server.registry.get_or_load("agentix_closures.race_pkg"),
    )
    assert results[0] is results[1] is results[2]
    # register() called exactly once across all three concurrent loaders.
    from agentix_closures.race_pkg import _register as race_reg
    assert race_reg._call_count == 1


# ── streaming ────────────────────────────────────────────────────


_STREAM_INIT = textwrap.dedent("""
    from dataclasses import dataclass
    from typing import AsyncIterator

    @dataclass
    class Token:
        text: str
        idx: int

    def chat(prompt: str, n: int = 3) -> AsyncIterator[Token]:
        raise NotImplementedError
""")

_STREAM_IMPL = textwrap.dedent("""
    from . import Token

    async def chat(prompt, n=3):
        for i in range(n):
            yield Token(text=f"{prompt}-{i}", idx=i)
""")

_STREAM_REGISTER = textwrap.dedent("""
    from agentix.dispatch import Dispatcher
    from . import chat
    from ._impl import chat as _chat
    def register():
        d = Dispatcher()
        d.bind(chat, _chat)
        return d
""")


async def test_dispatch_stream_yields_events(runtime_module, mount_package):
    """Dispatcher.dispatch_stream yields {item|end|error} dicts (transport-agnostic)."""
    from agentix.runtime.models import RemoteRequest

    server, _, _ = runtime_module
    mount_package(
        "streamer",
        package="agentix_closures.streamer",
        init_src=_STREAM_INIT,
        impl_src=_STREAM_IMPL,
        register_src=_STREAM_REGISTER,
    )
    await server._auto_load()
    dispatcher = await server.registry.get_or_load("agentix_closures.streamer")
    events = []
    async for ev in dispatcher.dispatch_stream(
        RemoteRequest(package="agentix_closures.streamer", method="chat",
                      kwargs={"prompt": "hi", "n": 2}),
    ):
        events.append(ev)
    assert events == [
        {"item": {"text": "hi-0", "idx": 0}},
        {"item": {"text": "hi-1", "idx": 1}},
        {"end": True},
    ]


async def test_stream_e2e_via_socketio(runtime_module, mount_package, live_server):
    """Full e2e: RuntimeClient.remote(stream_fn, ...) over a real Socket.IO conn."""
    mount_package(
        "streamer2",
        package="agentix_closures.streamer2",
        init_src=_STREAM_INIT,
        impl_src=_STREAM_IMPL,
        register_src=_STREAM_REGISTER,
    )
    base_url = await live_server()
    from agentix_closures.streamer2 import Token, chat

    async with RuntimeClient(base_url) as c:
        out = [tok async for tok in c.remote(chat, "go", n=3)]
    assert len(out) == 3
    assert all(isinstance(t, Token) for t in out)
    assert [t.text for t in out] == ["go-0", "go-1", "go-2"]


async def test_stream_impl_raises_mid_stream(runtime_module, mount_package, live_server):
    """Impl raises mid-stream → items observed first, then RemoteCallError."""
    init_src = textwrap.dedent("""
        from typing import AsyncIterator
        def explode(n: int = 2) -> AsyncIterator[int]:
            raise NotImplementedError
    """)
    impl_src = textwrap.dedent("""
        async def explode(n=2):
            for i in range(n):
                yield i
            raise ValueError("kaboom-after-yield")
    """)
    register_src = textwrap.dedent("""
        from agentix.dispatch import Dispatcher
        from . import explode
        from ._impl import explode as _e
        def register():
            d = Dispatcher(); d.bind(explode, _e); return d
    """)
    mount_package(
        "exploder",
        package="agentix_closures.exploder",
        init_src=init_src, impl_src=impl_src, register_src=register_src,
    )
    base_url = await live_server()
    from agentix_closures.exploder import explode

    collected: list[int] = []
    async with RuntimeClient(base_url) as c:
        with pytest.raises(RemoteCallError) as ei:
            async for x in c.remote(explode, n=2):
                collected.append(x)
    assert collected == [0, 1]
    assert ei.value.error.type == "ValueError"
    assert "kaboom-after-yield" in ei.value.error.message


# ── bidirectional ────────────────────────────────────────────────


_BIDI_INIT = textwrap.dedent("""
    from dataclasses import dataclass
    from typing import AsyncIterator

    @dataclass
    class UserMsg:
        text: str

    @dataclass
    class ReplyMsg:
        text: str

    def chat(messages: AsyncIterator[UserMsg], prefix: str = "say:") -> AsyncIterator[ReplyMsg]:
        raise NotImplementedError
""")

_BIDI_IMPL = textwrap.dedent("""
    from . import UserMsg, ReplyMsg

    async def chat(messages, prefix="say:"):
        async for m in messages:
            yield ReplyMsg(text=f"{prefix}{m.text}")
""")

_BIDI_REGISTER = textwrap.dedent("""
    from agentix.dispatch import Dispatcher
    from . import chat
    from ._impl import chat as _chat
    def register():
        d = Dispatcher(); d.bind(chat, _chat); return d
""")


async def test_bidi_e2e_via_socketio(runtime_module, mount_package, live_server):
    """Full e2e bidi: caller feeds AsyncIterator inputs, receives outputs."""
    mount_package(
        "chatter",
        package="agentix_closures.chatter",
        init_src=_BIDI_INIT, impl_src=_BIDI_IMPL, register_src=_BIDI_REGISTER,
    )
    base_url = await live_server()
    from agentix_closures.chatter import ReplyMsg, UserMsg, chat

    async def inputs():
        for s in ("hi", "bye", "."):
            yield UserMsg(text=s)

    async with RuntimeClient(base_url) as c:
        out = [r async for r in c.remote(chat, inputs(), prefix=">> ")]

    assert all(isinstance(r, ReplyMsg) for r in out)
    assert [r.text for r in out] == [">> hi", ">> bye", ">> ."]


# ── logs ─────────────────────────────────────────────────────────


async def test_logs_subscription(runtime_module, mount_echo, live_server):
    """Subscribing to logs receives records emitted by the runtime."""
    import logging as _logging

    mount_echo()
    base_url = await live_server()

    async with RuntimeClient(base_url) as c:
        records = []

        async def _collect():
            async for r in c.logs():
                records.append(r)
                if len(records) >= 1:
                    return

        collector = asyncio.create_task(_collect())
        await asyncio.sleep(0.5)  # let the subscription land server-side
        # Emit several times in case the first beats the subscription handshake.
        for _ in range(5):
            _logging.getLogger("agentix.test").warning("hello from the test")
            await asyncio.sleep(0.1)
        try:
            await asyncio.wait_for(collector, timeout=5)
        except TimeoutError:
            collector.cancel()
            raise

    assert any("hello from the test" in r.message for r in records)
    assert all(r.name.startswith("agentix") for r in records)


# ── support fixture ──────────────────────────────────────────────


@pytest.fixture
def mount_root_setup(runtime_module):
    """No-op: just ensures runtime_module ran (mount_root exists)."""
    return runtime_module[1]
