"""Tests for the eval-web dashboard server (no Docker)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from eval_web.app import create_app
from eval_web.run import stream_demo_run
from fastapi.testclient import TestClient


def test_index_is_served() -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Agentix" in resp.text
        assert "/ws/run" in resp.text  # the dashboard wires the run socket


def test_catalog_api_lists_agentix_packages() -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/api/catalog")
        assert resp.status_code == 200
        rows = resp.json()
        assert any(r["name"].startswith("agentix") for r in rows)
        assert all({"name", "kind", "version", "detail"} <= r.keys() for r in rows)


def test_ws_run_streams_start_results_done() -> None:
    with TestClient(create_app()) as client, client.websocket_connect("/ws/run?n=4&concurrency=2") as ws:
        types: list[str] = []
        results = 0
        while True:
            msg = ws.receive_json()
            types.append(msg["type"])
            if msg["type"] == "result":
                results += 1
            if msg["type"] == "done":
                assert msg["total"] == 4
                assert msg["resolved"] + msg["failed"] == 4
                break
        assert types[0] == "start"
        assert results == 4


async def test_stream_demo_run_emits_lifecycle_events() -> None:
    events: list[dict[str, Any]] = []

    async def send(event: dict[str, Any]) -> None:
        events.append(event)

    await stream_demo_run(send, n=5, n_concurrent=3, dur_scale=0.0)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert sum(1 for k in kinds if k == "result") == 5
    assert events[0]["instances"] and len(events[0]["instances"]) == 5
    assert "phase" in kinds  # at least one phase transition was reported


async def test_send_failure_tears_down_the_run() -> None:
    # If the consumer (a disconnected WebSocket) raises, the run must be torn
    # down and the error propagated — not hang, and not leak a background task.
    async def send(event: dict[str, Any]) -> None:
        if event["type"] == "phase":
            raise RuntimeError("client gone")

    with pytest.raises(RuntimeError, match="client gone"):
        await asyncio.wait_for(
            stream_demo_run(send, n=12, n_concurrent=3, dur_scale=0.05),
            timeout=5,
        )
