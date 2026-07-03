"""`Recorder` — host-side rollout capture at the tunnel.

Wrapping a client records every (request, response) pair its handlers
serve to a JSONL file, without the agent or the upstream noticing. The
tunnel is the one place all of an agent's LLM traffic passes, so this is
the natural recording point for rollout data collection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agentix.bridge import AbridgeError, ClientResponse, Recorder, Request, on


class _EchoClient:
    def __init__(self) -> None:
        self.closed = False

    @on("/v1/messages")
    async def messages(self, request: Request) -> ClientResponse:
        return ClientResponse.json({"echo": request.body["msg"], "id": "resp-1"})

    @on("/v1/messages/count_tokens")
    async def count_tokens(self, request: Request) -> ClientResponse:
        return ClientResponse.json({"input_tokens": 7})

    async def aclose(self) -> None:
        self.closed = True


class _FailingClient:
    @on("/v1/messages")
    async def messages(self, request: Request) -> ClientResponse:
        raise AbridgeError("upstream exploded", status_code=502)


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.asyncio
async def test_recorder_exposes_inner_routes_and_records_pairs(tmp_path) -> None:
    out = tmp_path / "run.jsonl"
    recorder = Recorder(_EchoClient(), out)
    routes = recorder.abridge_routes()
    assert set(routes) == {"/v1/messages", "/v1/messages/count_tokens"}

    resp = await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "hi"}))
    assert json.loads(resp.body) == {"echo": "hi", "id": "resp-1"}

    (record,) = _lines(out)
    assert record["path"] == "/v1/messages"
    assert record["request"] == {"msg": "hi"}
    assert record["response"]["body"] == {"echo": "hi", "id": "resp-1"}
    assert record["response"]["status_code"] == 200
    assert "ts" in record


@pytest.mark.asyncio
async def test_recorder_appends_one_line_per_call_in_order(tmp_path) -> None:
    out = tmp_path / "run.jsonl"
    recorder = Recorder(_EchoClient(), out)
    routes = recorder.abridge_routes()
    for i in range(3):
        await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": f"m{i}"}))
    assert [r["request"]["msg"] for r in _lines(out)] == ["m0", "m1", "m2"]


@pytest.mark.asyncio
async def test_recorder_records_handler_errors_and_reraises(tmp_path) -> None:
    out = tmp_path / "run.jsonl"
    recorder = Recorder(_FailingClient(), out)
    routes = recorder.abridge_routes()
    with pytest.raises(AbridgeError):
        await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "boom"}))

    (record,) = _lines(out)
    assert record["request"] == {"msg": "boom"}
    assert "upstream exploded" in record["error"]
    assert "response" not in record


@pytest.mark.asyncio
async def test_recorder_aclose_closes_inner_and_flushes(tmp_path) -> None:
    out = tmp_path / "run.jsonl"
    inner = _EchoClient()
    recorder = Recorder(inner, out)
    routes = recorder.abridge_routes()
    await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "x"}))
    await recorder.aclose()
    assert inner.closed
    assert len(_lines(out)) == 1


@pytest.mark.asyncio
async def test_recorder_preserves_non_json_bodies_as_text(tmp_path) -> None:
    class _SseClient:
        @on("/v1/messages")
        async def messages(self, request: Request) -> ClientResponse:
            return ClientResponse.sse(b"event: ping\ndata: {}\n\n")

    out = tmp_path / "run.jsonl"
    routes = Recorder(_SseClient(), out).abridge_routes()
    await routes["/v1/messages"](Request(path="/v1/messages", body={}))
    (record,) = _lines(out)
    assert record["response"]["media_type"] == "text/event-stream"
    assert record["response"]["body"] == "event: ping\ndata: {}\n\n"


def test_recorder_delegates_environ(tmp_path) -> None:
    class _EnvClient(_EchoClient):
        def environ(self, handle) -> dict[str, str]:
            return {"X": "y"}

    recorder = Recorder(_EnvClient(), tmp_path / "run.jsonl")
    assert recorder.environ(None) == {"X": "y"}
