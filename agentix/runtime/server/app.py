"""Agentix runtime server.

Multiplexes dispatch to per-namespace worker subprocesses. The runtime
process itself doesn't load namespace code — each namespace runs in its
own venv's Python, isolated from every other namespace's deps.

Endpoints:

- `POST /_remote` — typed **unary** dispatch (one request → one response)
- Socket.IO at `/socket.io/` — server-streaming, bidi, and log
  subscription, multiplexed by `call_id` on a single connection
- `GET /namespaces` — inventory (cheap; doesn't spawn workers)
- `GET /health`

Discovery walks `importlib.metadata.entry_points(group="agentix.namespace")`
in the runtime's own venv; in a bundle image, each namespace's venv has
been pip-installed alongside the runtime so this finds them all. Worker
spawning is lazy — `python -m agentix.runtime.worker --target ...` runs
on the first `/_remote` call for that namespace.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from agentix import __version__
from agentix.runtime.codec import pack, unpack
from agentix.runtime.models import (
    HealthResponse,
    NamespaceInfo,
    RemoteRequest,
)
from agentix.runtime.multiplexer import NamespaceMultiplexer
from agentix.runtime.server.llm_proxy import router as llm_proxy_router
from agentix.runtime.server.sio import make_sio
from agentix.runtime.server.trace_bridge import install_trace_bridge

logger = logging.getLogger("agentix.runtime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    multiplexer: NamespaceMultiplexer = app.state.multiplexer
    multiplexer.discover_entry_points()
    try:
        yield
    finally:
        await multiplexer.shutdown()


# Multiplexer is constructed here so tests can replace it via app.state
# before the lifespan kicks in. The trace forwarder is bound when the
# SIO server is attached (see bottom of file).
_multiplexer = NamespaceMultiplexer()

_fastapi_app = FastAPI(title="agentix", version=__version__, lifespan=lifespan)
_fastapi_app.state.multiplexer = _multiplexer
_fastapi_app.include_router(llm_proxy_router)


# ── Health & inventory ──────────────────────────────────────────


@_fastapi_app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@_fastapi_app.get("/namespaces")
async def list_namespaces() -> list[NamespaceInfo]:
    """All discovered namespaces. Cheap — doesn't spawn workers."""
    multiplexer: NamespaceMultiplexer = _fastapi_app.state.multiplexer
    return [NamespaceInfo(manifest=m) for m in multiplexer.manifests()]


# ── Remote dispatch ─────────────────────────────────────────────


@_fastapi_app.post("/_remote")
async def remote_call(request: Request) -> Response:
    """Unary dispatch endpoint. Spawns the worker on first call.

    Body: msgpack-encoded `{"package", "method", "args", "kwargs", "call_id"}`.
    Response: msgpack-encoded RemoteResponse dict. Always 200 — error info
    lives inside the response body (`{"ok": false, "error": {...}}`).
    Only "namespace not loaded" returns 404.

    Streaming and bidirectional methods are NOT served here — they
    live on the Socket.IO connection at `/socket.io/`. The worker
    itself rejects mismatches via the wire-pattern errors returned in
    `RemoteResponse.error`.
    """
    body = await request.body()
    raw = unpack(body)
    req = RemoteRequest.model_validate(raw)
    multiplexer: NamespaceMultiplexer = _fastapi_app.state.multiplexer
    if not multiplexer.has(req.package):
        raise HTTPException(
            status_code=404,
            detail=f"namespace not loaded: package={req.package!r}",
        )
    resp = await multiplexer.dispatch_unary(req)
    return Response(content=pack(resp.model_dump(mode="python")),
                    media_type="application/msgpack")


# ── Compose ASGI app: FastAPI for HTTP, Socket.IO for streams/logs ──
#
# The combined ASGI app is what uvicorn (and tests) run as
# `agentix.runtime.server:app`. `socketio.ASGIApp` routes `/socket.io/*`
# to the Socket.IO server and everything else to FastAPI.

import socketio as _socketio  # noqa: E402

_sio, _ = make_sio(_multiplexer)
app = _socketio.ASGIApp(_sio, _fastapi_app, socketio_path="/socket.io")
app.fastapi = _fastapi_app  # type: ignore[attr-defined]
app.state = _fastapi_app.state  # type: ignore[attr-defined]
_fastapi_app.state.sio = _sio
app.sio = _sio  # type: ignore[attr-defined]

# Trace bridge: every trace event from a namespace (in-process or
# subprocess worker) flows here, which forwards to the Socket.IO `traces`
# room. Wire it both ways: as an in-process trace sink (catches events
# from in-process workers in tests) AND as the multiplexer's worker
# trace_forwarder (catches subprocess worker frames in production).
_trace_forwarder = install_trace_bridge(_sio)
_multiplexer._trace_forwarder = _trace_forwarder  # type: ignore[attr-defined]


# ── Entry point (the bundle image's Docker ENTRYPOINT) ─────────


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

    uvicorn.run("agentix.runtime.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
