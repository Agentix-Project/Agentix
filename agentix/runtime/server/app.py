"""Agentix runtime server.

Runs remote calls through one runtime worker subprocess.

Endpoints:

- `GET /health`
- `POST /call` — internal fast-path used by `RuntimeClient.remote`;
  msgpack request/response. The caller sends RFC 7240
  `Prefer: respond-async, wait=N` (N seconds; fractional accepted).
  Returns **200** with the result if it lands within that budget;
  otherwise **202** with `{call_id}` + a `Location` header, and the
  result follows on Socket.IO (`call:result` / `call:error`). The
  honored budget is echoed in `Preference-Applied: wait=N`.
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
import re
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


_MSGPACK = "application/msgpack"
# RFC 7240 `wait` is delta-seconds (integer); we accept fractional too
# since we own both ends — a benign superset that keeps sub-second
# budgets expressible.
_WAIT_RE = re.compile(r"(?:^|[,;\s])wait\s*=\s*\"?([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_DEFAULT_WAIT_S = 1.0


def _parse_prefer_wait(prefer: str, *, default: float) -> float:
    match = _WAIT_RE.search(prefer or "")
    if not match:
        return default
    try:
        return max(float(match.group(1)), 0.0)
    except ValueError:
        return default


def _format_wait(seconds: float) -> str:
    return str(int(seconds)) if seconds == int(seconds) else str(seconds)


@_fastapi_app.post("/call")
async def call(request: Request) -> Response:
    """Internal fast-path endpoint used by `RuntimeClient.remote`.

    Request/response payloads are msgpack bytes, not JSON. The sync
    budget is the RFC 7240 `Prefer: respond-async, wait=N` header.
    """
    raw = await request.body()
    payload = unpack(raw) if raw else {}
    if not isinstance(payload, dict):
        payload = {}

    if not isinstance(payload.get("call_id"), str):
        error = {"type": "BadRequest", "message": "missing or invalid call_id"}
        return Response(
            content=pack({"ok": False, "error": error}),
            status_code=400,
            media_type=_MSGPACK,
        )

    wait_s = _parse_prefer_wait(request.headers.get("prefer", ""), default=_DEFAULT_WAIT_S)
    applied = {"Preference-Applied": f"wait={_format_wait(wait_s)}"}

    submit = getattr(_sio, "submit_http_call")
    result = await submit(payload, wait_s=wait_s)

    if result.get("accepted") is True:
        call_id = result.get("call_id")
        return Response(
            content=pack({"call_id": call_id}),
            status_code=202,
            media_type=_MSGPACK,
            headers={**applied, "Location": f"/call/{call_id}"},
        )

    # Completed within the budget (200), success or remote exception —
    # both are a delivered RPC outcome, distinguished by the `ok` flag.
    body = {key: result[key] for key in ("ok", "value", "error") if key in result}
    return Response(content=pack(body), status_code=200, media_type=_MSGPACK, headers=applied)


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
