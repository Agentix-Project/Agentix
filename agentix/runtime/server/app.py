"""Agentix runtime server.

In-process closure dispatch. The runtime is a single Python process serving:

- built-in operations (exec/upload/download) mounted at root
- `POST /_remote` — typed **unary** dispatch (one request → one response)
- Socket.IO at `/socket.io/` — server-streaming, bidi, and log subscription,
  multiplexed by `call_id` on a single connection
- `GET /closures` — inventory (always cheap; does not force-load anything)
- `GET /health`

There are no caller-chosen namespaces: each closure's Python import path
(`manifest.package`) is its routing key. Two images shipping the same
package collide; the second is skipped with a warning.

Discovery on startup is cheap: scan `/mnt/*` for `entry/manifest.json`,
validate against ClosureManifest, prepend each closure's `entry/python`
to sys.path, and register the closure in the Registry as a pending entry.
The actual `importlib.import_module(<pkg>)` + `_register.register()` is
deferred until the first call for that package — slow boots no longer
block the runtime, and a broken closure does not crash startup.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from agentix import __version__
from agentix.dispatch import Registry
from agentix.models import AGENTIX_CLOSURE_ABI, ClosureManifest
from agentix.runtime.models import (
    ClosureInfo,
    HealthResponse,
    RemoteRequest,
    RemoteResponse,
)
from agentix.runtime.server.builtins import router as builtins_router
from agentix.runtime.server.sio import make_sio

logger = logging.getLogger("agentix.runtime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

CLOSURE_MOUNT_ROOT = Path(os.environ.get("AGENTIX_CLOSURE_MOUNT_ROOT", "/mnt"))

registry = Registry()


async def _auto_load() -> None:
    """Scan /mnt for closures and register each one as a pending entry.

    Does NOT import any closure packages or build any Dispatchers — that
    work happens lazily on first `/_remote` call (see `Registry.get_or_load`).
    Cheap manifest validation is sufficient at boot to fail loudly on a
    malformed mount while keeping startup latency flat.

    `/mnt/runtime` is reserved (the runtime itself) and skipped.
    """
    if not CLOSURE_MOUNT_ROOT.is_dir():
        return
    for mount in sorted(CLOSURE_MOUNT_ROOT.iterdir()):
        if mount.name == "runtime" or not mount.is_dir():
            continue
        manifest = _read_manifest(mount)
        if manifest is None:
            continue
        if manifest.package in registry:
            logger.error(
                "skip mount %s: package %r already registered from %s",
                mount, manifest.package, registry.mount_for(manifest.package),
            )
            continue
        registry.register(manifest.package, manifest, mount)
        logger.info("registered closure '%s' from %s (deferred)",
                    manifest.package, mount)


def _read_manifest(mount: Path) -> ClosureManifest | None:
    """Read and validate <mount>/entry/manifest.json."""
    mf_path = mount / "entry" / "manifest.json"
    if not mf_path.is_file():
        logger.warning("skip mount %s: missing entry/manifest.json", mount)
        return None
    try:
        manifest = ClosureManifest.model_validate_json(mf_path.read_text())
    except ValidationError as exc:
        logger.error("skip mount %s: invalid manifest.json: %s", mount, exc)
        return None
    if manifest.abi != AGENTIX_CLOSURE_ABI:
        logger.warning(
            "skip mount %s: abi=%d, runtime supports %d",
            mount, manifest.abi, AGENTIX_CLOSURE_ABI,
        )
        return None
    return manifest


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _auto_load()
    yield


app = FastAPI(title="agentix", version=__version__, lifespan=lifespan)
app.state.registry = registry
app.include_router(builtins_router)


# ── Health & inventory ──────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@app.get("/closures")
async def list_closures() -> list[ClosureInfo]:
    """All registered closures (loaded or not). Doesn't force-load."""
    out: list[ClosureInfo] = []
    for pkg in registry.packages():
        manifest = registry.manifest_for(pkg)
        mount = registry.mount_for(pkg)
        if manifest is None or mount is None:
            continue
        out.append(ClosureInfo(path=str(mount), manifest=manifest))
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

import asyncio as _asyncio  # noqa: E402

import socketio as _socketio  # noqa: E402

import agentix.trace as _trace  # noqa: E402
from agentix.runtime.models import TraceEvent as _TraceEvent  # noqa: E402

_sio, _ = make_sio(registry)
_fastapi_app = app  # the FastAPI instance built above
app = _socketio.ASGIApp(_sio, _fastapi_app, socketio_path="/socket.io")
# Re-expose attributes that tests / extensions reach for via `server.app.*`.
app.fastapi = _fastapi_app  # type: ignore[attr-defined]
app.state = _fastapi_app.state  # type: ignore[attr-defined]
_fastapi_app.state.sio = _sio
app.sio = _sio  # type: ignore[attr-defined]


# Wire the closure-side `agentix.trace.emit(...)` helper to the Socket.IO
# `trace` room. Emission is best-effort and never blocks the caller — sync
# logging-style fire-and-forget. The dispatcher pins call_id / source via
# contextvars, so closure impls just write `trace.emit("kind", {...})`.

def _trace_emitter(
    kind: str,
    payload: dict,
    call_id: str | None,
    source: str | None,
) -> None:
    event = _TraceEvent(
        kind=kind, payload=payload, timestamp=_trace.now(),
        call_id=call_id, source=source,
    )
    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_sio.emit("trace", event.model_dump(mode="json"), room="traces"))


_trace._install_emitter(_trace_emitter)


# ── Entry point (invoked as /mnt/runtime/entry/bin/start) ───────


def main() -> None:
    """Entry point the closure convention expects at
    /mnt/runtime/entry/bin/start. Port via AGENTIX_BIND_PORT (env, default
    8000); dev shell can override via --port.
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
