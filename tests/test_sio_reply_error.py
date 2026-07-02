"""`Namespace._on_reply_error` threads the error envelope's `status_code`
onto `RemoteSioError`, so protocol layers built on `Namespace.request` (e.g.
abridge's sandbox tunnel) can reply with the real upstream status instead of
collapsing every remote failure to one blanket code."""

from __future__ import annotations

import asyncio

import pytest

from agentix.sio import Namespace, RemoteSioError


class _Ns(Namespace):
    namespace = "/reply-error-test"


async def test_reply_error_carries_status_code() -> None:
    ns = _Ns()
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    ns._pending_requests["r1"] = fut

    await ns._on_reply_error({
        "request_id": "r1",
        "error": {"type": "AbridgeError", "message": "rate limited", "status_code": 429},
    })

    with pytest.raises(RemoteSioError) as ei:
        fut.result()
    assert ei.value.status_code == 429
    assert ei.value.type == "AbridgeError"
    assert ei.value.message == "rate limited"


async def test_reply_error_non_int_status_code_is_none() -> None:
    ns = _Ns()
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    ns._pending_requests["r2"] = fut

    await ns._on_reply_error({
        "request_id": "r2",
        "error": {"type": "Boom", "message": "no status", "status_code": "429"},
    })

    with pytest.raises(RemoteSioError) as ei:
        fut.result()
    assert ei.value.status_code is None
