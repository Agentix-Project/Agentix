"""Agentix runtime server.

Runs remote calls through one runtime worker subprocess.

Endpoints:

- `GET /health`
- `POST /call` — internal fast-path used by `RuntimeClient.remote`;
  msgpack request/response. Returns the result inline if it lands
  within the caller's `prefer_sync_ms` budget; otherwise returns
  `accepted` and the result follows on Socket.IO.
- Socket.IO at `/socket.io/` — unary RPC on `/rpc` (`call` / `call:result` /
  `call:error`, `cancel`, plus `resume`/`ack` for reconnect-safe
  delivery), and side-channel namespaces (`/trace`, `/log`, and
  plugin paths registered via `agentix.sio`).

Remote requests carry a `RemoteCallable` import path plus a pickle of
the (args, kwargs) tuple. Only importable top-level functions and
builtins are supported call targets.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from agentix import __version__
from agentix.runtime.server.sio import make_sio
from agentix.runtime.server.worker import RuntimeWorkerClient
from agentix.runtime.shared.codec import pack, unpack
from agentix.runtime.shared.models import HealthResponse
from agentix.utils.log import configure_logging

configure_logging(default_context="sandbox-{uname}")
logger = logging.getLogger("agentix.runtime")


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker: RuntimeWorkerClient = app.state.worker
    try:
        yield
    finally:
        await worker.shutdown()


# Worker client is constructed here so tests can replace it via app.state
# before the lifespan kicks in.
_worker = RuntimeWorkerClient()

_fastapi_app = FastAPI(title="agentix", version=__version__, lifespan=lifespan)
_fastapi_app.state.worker = _worker


# ── Health & inventory ──────────────────────────────────────────


@_fastapi_app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@_fastapi_app.post("/call")
async def call(request: Request) -> Response:
    """Internal fast-path endpoint used by `RuntimeClient.remote`.

    Request/response payloads are msgpack bytes, not JSON.
    """
    raw = await request.body()
    payload = unpack(raw) if raw else {}
    if not isinstance(payload, dict):
        payload = {}

    prefer_sync_ms = payload.get("prefer_sync_ms")
    if not isinstance(prefer_sync_ms, int):
        prefer_sync_ms = 1000

    submit = getattr(_sio, "submit_http_call")
    result = await submit(payload, prefer_sync_ms=prefer_sync_ms)
    return Response(content=pack(result), media_type="application/msgpack")


# ── Compose ASGI app: FastAPI health + Socket.IO remote calls ──
#
# The combined ASGI app is what uvicorn runs as
# `agentix.runtime.server:app`. `socketio.ASGIApp` routes `/socket.io/*`
# to the Socket.IO server and everything else to FastAPI.

import socketio as _socketio  # noqa: E402

_sio, _ = make_sio(_worker)
app = _socketio.ASGIApp(_sio, _fastapi_app, socketio_path="/socket.io")
app.fastapi = _fastapi_app  # type: ignore[attr-defined]
app.state = _fastapi_app.state  # type: ignore[attr-defined]
app.sio = _sio  # type: ignore[attr-defined]

# The process entry point lives in `agentix.runtime.server.entrypoint`
# (invoked by the bundle's `/nix/runtime/bootstrap.sh` as `python -m`).
# This module is library-only — import `app` to mount the ASGI app in a
# different host (tests, embedded usage, ...) and run it however you
# want.
