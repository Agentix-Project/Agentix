"""Protocol-level integration tests for the runtime server's dispatch path.

These tests drive the real `agentix.runtime.server` FastAPI app over an
ASGI transport. There is no subprocess, no UDS, no reverse proxy — the
runtime imports each mounted closure's Python package in-process and
serves `POST /_remote` by calling the bound impl directly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import httpx
import pytest

from agentix import RemoteCallError, RuntimeClient
from agentix.models import RemoteRequest

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
        register_src="from agentix.dispatch import Dispatcher\nfrom . import x\nfrom ._impl import x as _x\ndef register():\n    d = Dispatcher(); d.bind(x, _x); return d",
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


# ── support fixture ──────────────────────────────────────────────


@pytest.fixture
def mount_root_setup(runtime_module):
    """No-op: just ensures runtime_module ran (mount_root exists)."""
    return runtime_module[1]
