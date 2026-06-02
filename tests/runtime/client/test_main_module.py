"""Remote calls can target regular importable modules.

`from my_app.tasks import fn; c.remote(fn, ...)` works when the module
is importable in the runtime venv.
"""

from __future__ import annotations

from agentix import RuntimeClient
from agentix.runtime.server.worker import RuntimeWorkerClient
from tests._rpc_helpers import request_for


async def test_remote_call_to_importable_module():
    """Remote call to `tests._user_app_target`."""
    mp = RuntimeWorkerClient()
    from tests._user_app_target import add, greet

    try:
        from agentix.runtime.shared.codec import unpack

        resp = await mp.call(request_for(greet, kwargs={"name": "world"}))
        assert resp.ok, resp.error
        assert unpack(resp.value) == "hello world"

        # Second function on the same module should reuse the same worker.
        resp2 = await mp.call(request_for(add, kwargs={"a": 3, "b": 4}))
        assert resp2.ok, resp2.error
        assert unpack(resp2.value) == 7

    finally:
        await mp.shutdown()


async def test_client_remote_accepts_imported_function(live_server):
    from tests._user_app_target import greet

    base_url = await live_server()
    async with RuntimeClient(base_url) as c:
        assert await c.remote(greet, name="world") == "hello world"
