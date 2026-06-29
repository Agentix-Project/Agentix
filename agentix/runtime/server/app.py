"""Agentix runtime server.

Runs remote calls through one runtime worker subprocess.

Endpoints:

- `GET /health`
- Socket.IO at `/socket.io/` — unary RPC on `/rpc` (`call` / `call:result` /
  `call:error`, `cancel`, plus `resume`/`ack` for reconnect-safe
  delivery), and side-channel namespaces (`/trace`, `/log`, and
  plugin paths registered via `agentix.sio`). Every `c.remote(...)`
  rides this one transport.

Remote requests carry a `RemoteCallable` import path plus a pickle of
the (args, kwargs) tuple. Only importable top-level functions and
builtins are supported call targets.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentix import __version__
from agentix.runtime.server.sio import make_sio
from agentix.runtime.server.worker import RuntimeWorkerClient
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


# ── Compose ASGI app: FastAPI health + Socket.IO remote calls ──
#
# The combined ASGI app is what uvicorn runs as
# `agentix.runtime.server:app`. `socketio.ASGIApp` routes `/socket.io/*`
# to the Socket.IO server and everything else to FastAPI.

import socketio as _socketio  # noqa: E402

_sio = make_sio(_worker)
app = _socketio.ASGIApp(_sio, _fastapi_app, socketio_path="/socket.io")
app.fastapi = _fastapi_app  # type: ignore[attr-defined]
app.state = _fastapi_app.state  # type: ignore[attr-defined]
app.sio = _sio  # type: ignore[attr-defined]

# The process entry point lives in `agentix.runtime.server.entrypoint`
# (invoked by the bundle's `/nix/runtime/bootstrap.sh` as `python -m`).
# This module is library-only — import `app` to mount the ASGI app in a
# different host (tests, embedded usage, ...) and run it however you
# want.
