"""Gateway node lifecycle — boots the dispatcher, serves the API, heartbeats.

A `GatewayNode` can register with a remote rollout server and send
periodic heartbeats, but the heartbeat target is optional — Agentix
gateways also run standalone (notebook driver, CI eval harness, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn

from agentix.deployment.base import Deployment
from agentix.gateway.completion_writer import CompletionWriter, NullCompletionWriter
from agentix.gateway.dispatcher import Dispatcher, ResultCallback
from agentix.gateway.server import build_app
from agentix.gateway.storage import RecordStore, SessionStore

logger = logging.getLogger("agentix.gateway.node")


@dataclass
class NodeConfig:
    """How the node identifies itself + optionally talks to a coordinator.

      * `node_id`     — stable id for this gateway process; defaults
                        to `<hostname>-<short-uuid>`.
      * `host`/`port` — where the FastAPI server binds.
      * `coordinator_url` — optional rollout-server URL to register
                            with + heartbeat to. When `None`, the
                            node runs standalone.
      * `heartbeat_interval_s` — how often to ping the coordinator
                                 (no-op when `coordinator_url=None`).
    """

    node_id: str = field(default_factory=lambda: f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}")
    host: str = "0.0.0.0"
    port: int = 8080
    coordinator_url: str | None = field(
        default_factory=lambda: os.environ.get("AGENTIX_GATEWAY_COORDINATOR_URL")
    )
    heartbeat_interval_s: float = 30.0
    concurrency: int = 4


class GatewayNode:
    """One gateway process: dispatcher + HTTP server + optional coordinator client.

    Use the `serve(...)` context manager for short-running scripts
    (notebooks, eval harnesses) — it boots uvicorn in the same loop
    and tears everything down on exit.

    Pass a `result_callback` (or set `coordinator_url`) when the
    gateway should hand `SessionResult`s back to a remote rollout
    server; without one, results live in the in-process
    `SessionStore` for polling.
    """

    def __init__(
        self,
        *,
        deployment: Deployment,
        config: NodeConfig | None = None,
        host_namespace_factory: Any = None,
        record_writer: CompletionWriter | None = None,
        sessions: SessionStore | None = None,
        records: RecordStore | None = None,
        result_callback: ResultCallback | None = None,
    ) -> None:
        self._config = config or NodeConfig()
        writer = record_writer or NullCompletionWriter()
        self._record_writer = writer
        store_records = records or RecordStore()
        store_sessions = sessions or SessionStore()
        self._records = store_records
        self._sessions = store_sessions

        forwarded_callback = result_callback
        if forwarded_callback is None and self._config.coordinator_url:
            forwarded_callback = _build_coordinator_callback(
                self._config.coordinator_url, self._config.node_id
            )

        self._dispatcher = Dispatcher(
            deployment=deployment,
            host_namespace_factory=host_namespace_factory,
            sessions=store_sessions,
            records=store_records,
            concurrency=self._config.concurrency,
            result_callback=forwarded_callback,
        )

    @property
    def config(self) -> NodeConfig:
        return self._config

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    @property
    def sessions(self) -> SessionStore:
        return self._sessions

    @property
    def records(self) -> RecordStore:
        return self._records

    @asynccontextmanager
    async def serve(self) -> AsyncIterator[GatewayNode]:
        """Boot the HTTP server (and the heartbeat, if configured)."""
        app = build_app(self._dispatcher, node_id=self._config.node_id)
        uvicorn_config = uvicorn.Config(
            app,
            host=self._config.host,
            port=self._config.port,
            log_level="info",
        )
        server = uvicorn.Server(uvicorn_config)
        server_task = asyncio.create_task(server.serve())
        heartbeat_task = (
            asyncio.create_task(self._heartbeat_loop())
            if self._config.coordinator_url
            else None
        )
        try:
            await _wait_uvicorn_started(server)
            yield self
        finally:
            server.should_exit = True
            await asyncio.gather(server_task, return_exceptions=True)
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            try:
                self._record_writer.close()
            except Exception:
                logger.exception("record_writer.close raised")

    async def _heartbeat_loop(self) -> None:
        url = f"{self._config.coordinator_url}/agentix/gateway/heartbeat"  # type: ignore[union-attr]
        while True:
            try:
                payload = {
                    "node_id": self._config.node_id,
                    "host": self._config.host,
                    "port": self._config.port,
                    "sessions": self._sessions.stats(),
                    "records": self._records.stats(),
                    "paused": self._dispatcher.paused,
                }
                async with httpx.AsyncClient(timeout=5) as client:
                    await client.post(url, json=payload)
            except Exception:
                logger.warning("heartbeat failed", exc_info=True)
            await asyncio.sleep(self._config.heartbeat_interval_s)


def _build_coordinator_callback(coordinator_url: str, node_id: str) -> ResultCallback:
    """Post a SessionResult back to the configured coordinator on terminal."""
    url = f"{coordinator_url}/agentix/gateway/session_result"

    async def _callback(result) -> None:  # type: ignore[no-untyped-def]
        try:
            payload = {
                "node_id": node_id,
                "session_id": result.session_id,
                "status": result.status.value,
                "started_at": result.started_at,
                "ended_at": result.ended_at,
                "duration_ms": result.duration_ms,
                "value": _safe_repr(result.value),
                "error": result.error,
                "records": result.records,
                "trajectory": result.trajectory,
                "metadata": result.metadata,
            }
            async with httpx.AsyncClient(timeout=15) as client:
                await client.post(url, json=payload)
        except Exception:
            logger.exception("coordinator callback failed")

    return _callback


def _safe_repr(value: Any) -> Any:
    """Stringify anything that doesn't JSON-serialize cleanly."""
    import json

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


async def _wait_uvicorn_started(server: uvicorn.Server) -> None:
    for _ in range(400):
        if getattr(server, "started", False):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("uvicorn did not bind in time")


__all__ = ["GatewayNode", "NodeConfig"]
