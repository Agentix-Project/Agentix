"""FastAPI app exposing the gateway's REST surface.

Endpoints (all JSON):

  POST /sessions                       dispatch a SessionSpec, returns Session
  GET  /sessions                       list live + terminal sessions
  GET  /sessions/{id}                  one session's current state
  GET  /sessions/{id}/result           SessionResult (404 until terminal)
  GET  /sessions/{id}/records          captured CompletionRecords for a session
  GET  /records                        all captured records on this node
  POST /pause                          stop new RUNNING transitions
  POST /resume                         resume new RUNNING transitions
  GET  /health                         liveness + heartbeat shape

The app is constructed by `build_app(dispatcher)` so tests can wire a
stub dispatcher without spinning a real Deployment.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from agentix.gateway.dispatcher import Dispatcher
from agentix.gateway.session import Session, SessionResult, SessionSpec


def build_app(dispatcher: Dispatcher, *, node_id: str | None = None) -> FastAPI:
    app = FastAPI(title="agentix-gateway")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "paused": dispatcher.paused,
            "node_id": node_id,
            "sessions": dispatcher.sessions.stats(),
            "records": dispatcher.records.stats(),
        }

    @app.post("/sessions", status_code=202)
    async def create_session(spec: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = _coerce_spec(spec)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        session = dispatcher.dispatch(parsed)
        return _serialise_session(session)

    @app.get("/sessions")
    async def list_sessions() -> dict[str, Any]:
        live = [_serialise_session(s) for s in dispatcher.sessions.list_live()]
        results = [
            _serialise_result(r) for r in dispatcher.sessions.list_results()
        ]
        return {"live": live, "results": results}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        session = dispatcher.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return _serialise_session(session)

    @app.get("/sessions/{session_id}/result")
    async def get_session_result(session_id: str) -> dict[str, Any]:
        session = dispatcher.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        if not session.status.terminal:
            raise HTTPException(status_code=409, detail=f"session {session.status.value}")
        return _serialise_result(session.to_result())

    @app.get("/sessions/{session_id}/records")
    async def get_session_records(session_id: str) -> dict[str, Any]:
        session = dispatcher.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "records": session.records}

    @app.get("/records")
    async def get_records() -> dict[str, Any]:
        return {"records": dispatcher.records.snapshot()}

    @app.post("/pause")
    async def pause() -> dict[str, Any]:
        dispatcher.pause()
        return {"paused": True}

    @app.post("/resume")
    async def resume() -> dict[str, Any]:
        dispatcher.resume()
        return {"paused": False}

    return app


# ── serialisation helpers ─────────────────────────────────────────────────


def _coerce_spec(payload: dict[str, Any]) -> SessionSpec:
    required = ("callable_ref", "image", "bundle")
    for key in required:
        if key not in payload:
            raise ValueError(f"missing required field {key!r}")
    return SessionSpec(
        callable_ref=str(payload["callable_ref"]),
        image=str(payload["image"]),
        bundle=str(payload["bundle"]),
        args=tuple(payload.get("args") or ()),
        kwargs=dict(payload.get("kwargs") or {}),
        platform=payload.get("platform"),
        env=payload.get("env"),
        metadata=dict(payload.get("metadata") or {}),
        session_id=payload.get("session_id"),
        upstream_model=payload.get("upstream_model"),
    )


def _serialise_session(s: Session) -> dict[str, Any]:
    return {
        "session_id": s.session_id,
        "status": s.status.value,
        "sandbox_id": s.sandbox_id,
        "runtime_url": s.runtime_url,
        "started_at": s.started_at,
        "ended_at": s.ended_at,
        "error": s.error,
        "stage_durations_ms": dict(s.stage_durations_ms),
        "metadata": dict(s.spec.metadata),
        "spec": {
            "callable_ref": s.spec.callable_ref,
            "image": s.spec.image,
            "bundle": s.spec.bundle,
            "platform": s.spec.platform,
            "metadata": dict(s.spec.metadata),
        },
    }


def _serialise_result(r: SessionResult) -> dict[str, Any]:
    out = asdict(r)
    out["status"] = r.status.value
    out["duration_ms"] = r.duration_ms
    # `value` may be arbitrary Python; let JSON encoders coerce it.
    return out


def _starlette_json_response(data: Any, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status_code)


__all__ = ["build_app"]
