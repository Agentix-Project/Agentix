"""Agentix runtime server.

In-process closure dispatch. The runtime is a single Python process serving:

- `POST /_remote` — typed **unary** dispatch (one request → one response)
- Socket.IO at `/socket.io/` — server-streaming, bidi, and log subscription,
  multiplexed by `call_id` on a single connection
- `GET /closures` — inventory (always cheap; does not force-load anything)
- `GET /health`

Closure discovery uses `importlib.metadata.entry_points(group="agentix.closure")`:
the runtime walks every installed distribution's `[project.entry-points]`,
registers the closure as a pending entry, and defers `ep.load()` (the actual
import) until the first `/_remote` call for that closure. A broken closure
surfaces on call, not at boot. There's no on-disk `manifest.json`, no
`/mnt/<closure>` mount convention, no kind-segment namespace.

The closure's Python import path (e.g. `agentix.bash`) is the routing key;
there are no caller-chosen namespaces.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from agentix import __version__
from agentix import trace
from agentix.dispatch import Registry, discover_entry_points
from agentix.models import ClosureManifest
from agentix.runtime.models import (
    ClosureInfo,
    HealthResponse,
    RemoteRequest,
    RemoteResponse,
)
from agentix.runtime.server.llm_proxy import router as llm_proxy_router
from agentix.runtime.server.sio import make_sio
from agentix.runtime.server.trace_bridge import install as install_trace_bridge

logger = logging.getLogger("agentix.runtime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

registry = Registry()


async def _auto_load() -> None:
    """Discover installed closures via entry points; register each lazily.

    Walks `importlib.metadata.entry_points(group="agentix.closure")`. Each
    such entry point has the form `<short> = "<package>:<class>"` declared
    in the closure's `pyproject.toml`. Discovery is cheap — we record the
    `EntryPoint` object but don't call `ep.load()` until the closure is
    first dispatched. A broken closure (import error, bad class) thus
    fails on call, not at boot.
    """
    for ep in discover_entry_points():
        try:
            registry.register_entry_point(ep)
        except ValueError as exc:
            # `register_entry_point` raises on duplicate package — common
            # when two installed dists claim the same import path. Log
            # and skip the second.
            logger.error("entry-point %r: %s", ep.name, exc)
            continue
        logger.info(
            "registered closure '%s' (deferred) — entry point %r",
            ep.value.split(":", 1)[0], ep.name,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _auto_load()
    # Hook every third-party trace sink registered under
    # `agentix.trace_sink`. Installer errors are logged + skipped so
    # one broken sink doesn't block the runtime.
    trace.install_entry_point_sinks()
    yield


app = FastAPI(title="agentix", version=__version__, lifespan=lifespan)
app.state.registry = registry
app.include_router(llm_proxy_router)


# ── Health & inventory ──────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.get("/closures")
async def list_closures() -> list[ClosureInfo]:
    """All registered closures (loaded or not). Doesn't force-load."""
    out: list[ClosureInfo] = []
    for pkg in registry.packages():
        info = registry.info_for(pkg)
        if info is None:
            continue
        dist_name, dist_version = info
        out.append(ClosureInfo(manifest=ClosureManifest(
            name=dist_name or pkg.rsplit(".", 1)[-1],
            version=dist_version or "0.0.0",
            package=pkg,
        )))
    return out


# ── Remote dispatch ─────────────────────────────────────────────


@app.post("/_remote")
async def remote_call(request: RemoteRequest) -> RemoteResponse:
    """Unary dispatch endpoint. Triggers lazy import on first use.

    Streaming and bidirectional methods are NOT served here — they live on
    the Socket.IO connection at `/socket.io/`. A 400 with a hint is
    returned if the caller mistakenly POSTs a streaming method to /_remote.
    """
    try:
        dispatcher = await registry.get_or_load(request.package)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"closure '{request.package}' failed to load: {type(exc).__name__}: {exc}",
        ) from exc
    if dispatcher is None:
        raise HTTPException(
            status_code=404,
            detail=f"closure not loaded: package={request.package!r}",
        )
    if dispatcher.is_bidi(request.method):
        raise HTTPException(
            status_code=400,
            detail=(
                f"method '{request.method}' is bidirectional; "
                f"use the Socket.IO `bidi:start` event instead"
            ),
        )
    if dispatcher.is_streaming(request.method):
        raise HTTPException(
            status_code=400,
            detail=(
                f"method '{request.method}' returns AsyncIterator; "
                f"use the Socket.IO `stream` event instead"
            ),
        )
    return await dispatcher.dispatch(request)


# ── Compose ASGI app: FastAPI for HTTP, Socket.IO for streams/logs ──
#
# The combined ASGI app is what uvicorn (and tests) run as
# `agentix.runtime.server:app`. `socketio.ASGIApp` routes `/socket.io/*` to
# the Socket.IO server and everything else to FastAPI, so plain HTTP
# endpoints (`/health`, `/_remote`, …) work unchanged through ASGITransport.

import socketio as _socketio  # noqa: E402

_sio, _ = make_sio(registry)
_fastapi_app = app  # the FastAPI instance built above
app = _socketio.ASGIApp(_sio, _fastapi_app, socketio_path="/socket.io")
# Re-expose attributes that tests / extensions reach for via `server.app.*`.
app.fastapi = _fastapi_app  # type: ignore[attr-defined]
app.state = _fastapi_app.state  # type: ignore[attr-defined]
_fastapi_app.state.sio = _sio
app.sio = _sio  # type: ignore[attr-defined]

# Route closure-side `agentix.trace.emit(...)` into the Socket.IO `trace` room.
install_trace_bridge(_sio)


# ── Entry point (invoked as /mnt/runtime/entry/bin/start) ───────


def main() -> None:
    """Entry point exposed as the `start` console script. Port via
    AGENTIX_BIND_PORT (env, default 8000); dev shell can override via --port.
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
        import debugpy

        debugpy.listen(("0.0.0.0", args.debug_port))
        print(f"debugpy listening on 0.0.0.0:{args.debug_port}")
        if args.debug_wait:
            print("Waiting for debugger to attach...")
            debugpy.wait_for_client()

    uvicorn.run("agentix.runtime.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
