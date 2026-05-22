"""Protocol-level integration tests for remote calls.

Drives the runtime server over Socket.IO using the in-process worker
backend. Subprocess stdio is covered separately in
`test_worker_subprocess.py`.
"""

from __future__ import annotations

import asyncio
import functools

import httpx
import pytest
import socketio

from agentix import RemoteCallError, RuntimeClient
from agentix.runtime.shared.codec import pack, unpack
from agentix.runtime.shared.models import RemoteRequest
from tests import _worker_target as target
from tests._rpc_helpers import request_for

pytestmark = pytest.mark.asyncio


# ── basics ─────────────────────────────────────────────────────────────


async def test_http_remote_endpoint_is_not_registered(runtime_module):
    server, _, _ = runtime_module
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        r = await http.post("/_remote", content=b"")

    assert r.status_code == 404


async def test_socketio_call_serialized_callable(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    sio = socketio.AsyncClient()
    results: asyncio.Queue = asyncio.Queue()

    async def _on_result(data):
        await results.put(unpack(data))

    sio.on("call:result", _on_result)
    await sio.connect(base_url)
    try:
        req = request_for(target.echo, kwargs={"msg": "hi"}, call_id="call-ok")
        await sio.emit("call", pack(req.model_dump()))
        payload = await asyncio.wait_for(results.get(), timeout=5)
    finally:
        await sio.disconnect()

    assert payload["call_id"] == "call-ok"
    import pickle

    result = pickle.loads(payload["value"])
    assert result.msg == "echo:hi"


async def test_socketio_bad_callable_returns_error(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    sio = socketio.AsyncClient()
    errors: asyncio.Queue = asyncio.Queue()

    async def _on_error(data):
        await errors.put(unpack(data))

    sio.on("call:error", _on_error)
    await sio.connect(base_url)
    try:
        import pickle

        from agentix.runtime.shared.callables import RemoteCallable

        # Garbage import path that can't be resolved into a callable.
        req = RemoteRequest(
            callable=RemoteCallable("not-valid-import-path"),
            arguments=pickle.dumps(((), {})),
            call_id="call-bad",
        )
        await sio.emit("call", pack(req.model_dump()))
        payload = await asyncio.wait_for(errors.get(), timeout=5)
    finally:
        await sio.disconnect()

    assert payload["call_id"] == "call-bad"
    assert payload["error"]["type"] == "ValueError"


async def test_client_remote_round_trip(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        result = await c.remote(target.echo, msg="hello")
    assert result.msg == "echo:hello"


async def test_client_remote_large_payload(use_inprocess_worker, live_server):
    """A `c.remote` payload above the default 1 MB Socket.IO message cap
    must round-trip — not kill the websocket.

    Regression: Engine.IO's `max_http_buffer_size` (and the websocket
    libraries' own caps) default to ~1 MB. An RPC argument or an LLM
    request body easily exceeds that; before the caps were lifted the
    connection was dropped mid-call. 8 MB exercises well past 1 MB.
    """
    use_inprocess_worker()
    base_url = await live_server()
    blob = "x" * (8 * 1024 * 1024)
    async with RuntimeClient(base_url) as c:
        result = await c.remote(len, blob)
    assert result == len(blob)


async def test_client_remote_raises_on_impl_error(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        with pytest.raises(RemoteCallError):
            await c.remote(target.boom)


# ── seamless callable forms ────────────────────────────────────────────


async def test_remote_rejects_unimportable_lambda(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        # Lambdas do not have an importable top-level function path, so
        # the host-side `RemoteCallable._resolve(fn)` raises before the
        # call leaves.
        with pytest.raises(Exception):
            await c.remote(lambda x: x + 1, 41)


async def test_remote_rejects_partial(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    add_three = functools.partial(target.add, 3)
    async with RuntimeClient(base_url) as c:
        with pytest.raises(Exception):
            await c.remote(add_three, 4)


async def test_remote_rejects_bound_method(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        with pytest.raises(Exception):
            await c.remote(target.prefixer.bound, "hello")


async def test_remote_rejects_callable_instance(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        with pytest.raises(Exception):
            await c.remote(target.prefixer, "hello")


async def test_remote_accepts_script_main_function(
    use_inprocess_worker,
    live_server,
    tmp_path,
    monkeypatch,
):
    script = tmp_path / "runner_like.py"
    script.write_text(
        "async def get_patch(workdir):\n"
        "    return f'patch from {workdir}'\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    import __main__ as main_module

    monkeypatch.setattr(main_module, "__file__", str(script), raising=False)
    monkeypatch.setattr(main_module, "__spec__", None, raising=False)
    namespace = {"__name__": "__main__"}
    exec(
        "async def get_patch(workdir):\n"
        "    return f'patch from {workdir}'\n",
        namespace,
    )

    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        assert await c.remote(namespace["get_patch"], "/testbed") == "patch from /testbed"


# ── cancel ────────────────────────────────────────────────────────────


async def test_socketio_cancel_returns_cancelled_error(use_inprocess_worker, live_server):
    """Cancelling an in-flight call yields a Cancelled error."""
    use_inprocess_worker()
    base_url = await live_server()
    sio = socketio.AsyncClient()
    errors: asyncio.Queue = asyncio.Queue()

    async def _on_error(data):
        await errors.put(unpack(data))

    sio.on("call:error", _on_error)
    await sio.connect(base_url)
    try:
        # Use a slow remote call. asyncio.sleep is convenient — it's
        # importable and async; we just need it to outlast the cancel.
        import asyncio as _asyncio
        import pickle

        from agentix.runtime.shared.callables import RemoteCallable

        req = RemoteRequest(
            callable=RemoteCallable._resolve(_asyncio.sleep),
            arguments=pickle.dumps(((5.0,), {})),
            call_id="cancel-me",
        )
        await sio.emit("call", pack(req.model_dump()))
        await asyncio.sleep(0.1)
        await sio.emit("cancel", pack({"call_id": "cancel-me"}))
        payload = await asyncio.wait_for(errors.get(), timeout=5)
    finally:
        await sio.disconnect()

    assert payload["call_id"] == "cancel-me"
    assert payload["error"]["type"] == "Cancelled"
    assert payload["error"]["cancelled"] is True
