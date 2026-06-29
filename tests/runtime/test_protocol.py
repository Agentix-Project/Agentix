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

from agentix import Failed, Ok, RemoteCallError, RuntimeClient
from agentix.runtime.shared.codec import pack, unpack
from agentix.runtime.shared.models import RemoteRequest
from tests import _worker_target as target
from tests._rpc_helpers import request_for

pytestmark = pytest.mark.asyncio
RPC_NAMESPACE = "/rpc"


# ── basics ─────────────────────────────────────────────────────────────


async def test_http_rpc_endpoints_are_not_registered(runtime_module):
    """Only `/health` is served over HTTP — RPC has no HTTP endpoint;
    every `c.remote()` rides Socket.IO `/rpc`."""
    server, _, _ = runtime_module
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        assert (await http.post("/_remote", content=b"")).status_code == 404
        assert (await http.post("/call", content=b"")).status_code == 404


async def test_socketio_call_serialized_callable(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    sio = socketio.AsyncClient()
    results: asyncio.Queue = asyncio.Queue()

    async def _on_result(data):
        await results.put(unpack(data))

    sio.on("call:result", _on_result, namespace=RPC_NAMESPACE)
    await sio.connect(base_url, namespaces=[RPC_NAMESPACE])
    try:
        req = request_for(target.echo, kwargs={"msg": "hi"}, call_id="call-ok")
        await sio.emit("call", pack(req.model_dump()), namespace=RPC_NAMESPACE)
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

    sio.on("call:error", _on_error, namespace=RPC_NAMESPACE)
    await sio.connect(base_url, namespaces=[RPC_NAMESPACE])
    try:
        import pickle

        from agentix.runtime.shared.callables import RemoteCallable

        # Garbage import path that can't be resolved into a callable.
        req = RemoteRequest(
            callable=RemoteCallable("not-valid-import-path"),
            arguments=pickle.dumps(((), {})),
            call_id="call-bad",
        )
        await sio.emit("call", pack(req.model_dump()), namespace=RPC_NAMESPACE)
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


async def test_try_remote_returns_ok(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        result = await c.try_remote(target.add, 2, 3)
    assert isinstance(result, Ok)
    assert result.value == 5


async def test_try_remote_returns_failed(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        result = await c.try_remote(target.boom)
    assert isinstance(result, Failed)
    assert isinstance(result.error, RemoteCallError)


async def test_client_remote_long_call_round_trip(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        # A multi-second call round-trips over the single SIO transport.
        assert await c.remote(asyncio.sleep, 1.2) is None


async def test_same_call_id_runs_fn_exactly_once(use_inprocess_worker, live_server):
    """The runtime must execute `fn` exactly once per `call_id`, even
    when the same id arrives over every SIO submission path: a duplicate
    `call` and a `resume`.
    """
    use_inprocess_worker()
    base_url = await live_server()

    target._exec_counter = 0

    call_id = "once-only-1"
    req = request_for(target.count_exec_and_sleep, args=[0.6], call_id=call_id)
    payload_bytes = pack(req.model_dump())

    sio = socketio.AsyncClient()
    results: asyncio.Queue = asyncio.Queue()

    async def _on_result(data):
        await results.put(unpack(data))

    sio.on("call:result", _on_result, namespace=RPC_NAMESPACE)
    await sio.connect(base_url, namespaces=[RPC_NAMESPACE])
    try:
        # Same call_id submitted three times across the SIO paths.
        await sio.emit("call", payload_bytes, namespace=RPC_NAMESPACE)
        await sio.emit(
            "resume",
            pack({"call_ids": [call_id]}),
            namespace=RPC_NAMESPACE,
        )
        await sio.emit("call", payload_bytes, namespace=RPC_NAMESPACE)

        payload = await asyncio.wait_for(results.get(), timeout=5)
    finally:
        await sio.disconnect()

    assert payload["call_id"] == call_id
    import pickle as _pickle
    assert _pickle.loads(payload["value"]) == 1, "fn must have run exactly once"


async def test_runtime_replays_unacked_result_after_reconnect(use_inprocess_worker, live_server):
    """A task whose result arrives while the host is disconnected
    must be replayed on reconnect via the `resume` event, and `fn`
    must run exactly once across the whole flow.
    """
    use_inprocess_worker()
    base_url = await live_server()

    target._exec_counter = 0

    # First "session": submit a slow call, then drop the link before
    # the result has time to arrive.
    sio_a = socketio.AsyncClient()
    await sio_a.connect(base_url, namespaces=[RPC_NAMESPACE])
    call_id = "resume-test-1"
    req = request_for(
        target.count_exec_and_sleep,
        args=[0.6],
        call_id=call_id,
    )
    await sio_a.emit("call", pack(req.model_dump()), namespace=RPC_NAMESPACE)
    # Give the server time to register the in-flight task.
    await asyncio.sleep(0.1)
    await sio_a.disconnect()

    # Let the server task finish while nobody is connected.
    await asyncio.sleep(1.0)

    # Second "session": reconnect, ask the server to replay any results
    # the host is still waiting for, and confirm receipt.
    sio_b = socketio.AsyncClient()
    results: asyncio.Queue = asyncio.Queue()

    async def _on_result(data):
        await results.put(unpack(data))

    sio_b.on("call:result", _on_result, namespace=RPC_NAMESPACE)
    await sio_b.connect(base_url, namespaces=[RPC_NAMESPACE])
    try:
        await sio_b.emit(
            "resume",
            pack({"call_ids": [call_id]}),
            namespace=RPC_NAMESPACE,
        )
        payload = await asyncio.wait_for(results.get(), timeout=5)
    finally:
        await sio_b.disconnect()

    assert payload["call_id"] == call_id
    import pickle as _pickle
    assert _pickle.loads(payload["value"]) == 1, "fn must run exactly once"


async def test_resume_for_unknown_call_id_fails_definitively(use_inprocess_worker, live_server):
    """A `resume` for a call_id the runtime no longer holds (evicted
    under cap, or never seen) must return a definite `call:error` — the
    contract forbids silence, which would hang the host's `remote()`.
    """
    use_inprocess_worker()
    base_url = await live_server()

    sio = socketio.AsyncClient()
    errors: asyncio.Queue = asyncio.Queue()

    async def _on_error(data):
        await errors.put(unpack(data))

    sio.on("call:error", _on_error, namespace=RPC_NAMESPACE)
    await sio.connect(base_url, namespaces=[RPC_NAMESPACE])
    try:
        await sio.emit(
            "resume",
            pack({"call_ids": ["never-existed"]}),
            namespace=RPC_NAMESPACE,
        )
        payload = await asyncio.wait_for(errors.get(), timeout=5)
    finally:
        await sio.disconnect()

    assert payload["call_id"] == "never-existed"
    assert payload["error"]["type"] == "ResultUnavailable"


async def test_client_remote_runs_fn_exactly_once(use_inprocess_worker, live_server):
    use_inprocess_worker()
    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        await c.remote(target.reset_exec_counter)
        result = await c.remote(target.count_exec_and_sleep, 1.2)
    assert result == 1


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

    sio.on("call:error", _on_error, namespace=RPC_NAMESPACE)
    await sio.connect(base_url, namespaces=[RPC_NAMESPACE])
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
        await sio.emit("call", pack(req.model_dump()), namespace=RPC_NAMESPACE)
        await asyncio.sleep(0.1)
        await sio.emit(
            "cancel",
            pack({"call_id": "cancel-me"}),
            namespace=RPC_NAMESPACE,
        )
        payload = await asyncio.wait_for(errors.get(), timeout=5)
    finally:
        await sio.disconnect()

    assert payload["call_id"] == "cancel-me"
    assert payload["error"]["type"] == "Cancelled"
    assert payload["error"]["cancelled"] is True
