"""Agentix runtime server.

Runs remote calls through one runtime worker subprocess.

Endpoints:

- `GET /health`
- Socket.IO at `/socket.io/` — `call` / `call:result` / `call:error`,
  `cancel`, plus broadcast `trace:event`.

Remote requests carry a `RemoteCallable` import path plus
a pickle of the (args, kwargs) tuple. Module-level functions/classes,
bound methods, `functools.partial`, and pickleable callable instances
are the supported boundary.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentix import __version__
from agentix.runtime.server.sio import make_sio
from agentix.runtime.server.worker import RuntimeWorkerClient
from agentix.runtime.shared.models import HealthResponse

logger = logging.getLogger("agentix.runtime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


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

_sio, _ = make_sio(_worker)
app = _socketio.ASGIApp(_sio, _fastapi_app, socketio_path="/socket.io")
app.fastapi = _fastapi_app  # type: ignore[attr-defined]
app.state = _fastapi_app.state  # type: ignore[attr-defined]
app.sio = _sio  # type: ignore[attr-defined]


# ── Entry point (the bundle image's server command) ─────────


def main() -> None:
    """Entry point exposed as the `agentix-server` console script. Port
    via AGENTIX_BIND_PORT (env, default 8000); dev shell can override via
    --port.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="agentix runtime server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AGENTIX_BIND_PORT", "8000")),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-port", type=int, default=5678)
    parser.add_argument("--debug-wait", action="store_true")
    args = parser.parse_args()

    if args.debug:
        import debugpy  # type: ignore[reportMissingImports]

        debugpy.listen(("0.0.0.0", args.debug_port))
        print(f"debugpy listening on 0.0.0.0:{args.debug_port}")
        if args.debug_wait:
            print("Waiting for debugger to attach...")
            debugpy.wait_for_client()

    # `ws_max_size` lifts uvicorn's websocket frame cap to match the
    # Socket.IO layer's `max_http_buffer_size` — otherwise uvicorn would
    # be the bottleneck for a large `c.remote` payload / plugin event.
    from agentix.runtime.shared import MAX_MESSAGE_BYTES

    uvicorn.run(
        "agentix.runtime.server:app",
        host=args.host,
        port=args.port,
        ws_max_size=MAX_MESSAGE_BYTES,
    )


if __name__ == "__main__":
    main()
