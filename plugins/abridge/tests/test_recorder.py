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


@pytest.mark.asyncio
async def test_recorder_rows_carry_session_and_request_ids(tmp_path) -> None:
    """Rows are joinable against downstream token records: `session_id` (the
    rollout identity the Recorder was built with) and a per-call
    `request_id`, unique across calls."""
    out = tmp_path / "run.jsonl"
    recorder = Recorder(_EchoClient(), out, session_id="sess-42")
    routes = recorder.abridge_routes()
    await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "a"}))
    await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "b"}))

    first, second = _lines(out)
    assert first["session_id"] == second["session_id"] == "sess-42"
    assert first["request_id"] and second["request_id"]
    assert first["request_id"] != second["request_id"]


@pytest.mark.asyncio
async def test_recorder_request_id_matches_upstream_x_request_id(tmp_path, monkeypatch) -> None:
    """The alignment contract: the id in the Recorder row IS the
    `x-request-id` the transport stamps on the upstream hop (via the
    `current_request_id` context var), so a message-level row and a
    token-level sidecar record for the same call join on one key."""
    import httpx
    from agentix.bridge import Forward

    fwd = Forward("http://side.car", paths=["/v1/messages"], session_id="sess-1")
    seen_headers: list[dict] = []

    async def fake_post(url, *, json, headers):
        seen_headers.append(dict(headers))
        return httpx.Response(200, content=b'{"ok": true}', headers={"content-type": "application/json"})

    monkeypatch.setattr(fwd._client, "post", fake_post)

    out = tmp_path / "run.jsonl"
    routes = Recorder(fwd, out, session_id="sess-1").abridge_routes()
    await routes["/v1/messages"](Request(path="/v1/messages", body={"x": 1}))

    (row,) = _lines(out)
    assert seen_headers[0]["x-request-id"] == row["request_id"]
    assert seen_headers[0]["x-session-id"] == row["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_recorder_error_rows_also_carry_ids(tmp_path) -> None:
    out = tmp_path / "run.jsonl"
    routes = Recorder(_FailingClient(), out, session_id="sess-9").abridge_routes()
    with pytest.raises(AbridgeError):
        await routes["/v1/messages"](Request(path="/v1/messages", body={}))
    (row,) = _lines(out)
    assert row["session_id"] == "sess-9"
    assert row["request_id"]
    assert "error" in row


def test_recorder_opens_file_lazily(tmp_path) -> None:
    """A Recorder that never serves (e.g. build_session_app's route
    enumeration probe) must leave no empty file behind."""
    out = tmp_path / "probe.jsonl"
    recorder = Recorder(_EchoClient(), out)
    recorder.abridge_routes()
    assert not out.exists()


@pytest.mark.asyncio
async def test_recorder_write_failure_is_log_and_serve(tmp_path, monkeypatch, caplog) -> None:
    """Capture failures never fail the served call (matching the token
    gateway's policy): with the record path unwritable the agent still gets
    its response and the drop is logged."""
    import logging

    out = tmp_path / "run.jsonl"
    recorder = Recorder(_EchoClient(), out)
    routes = recorder.abridge_routes()

    def broken_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(out), "open", broken_open)
    with caplog.at_level(logging.ERROR, logger="agentix.bridge.recorder"):
        resp = await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "hi"}))
    assert json.loads(resp.body)["echo"] == "hi"  # the call succeeded
    assert any("row NOT persisted" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_recorder_drops_rows_after_aclose(tmp_path, caplog) -> None:
    """A straggler dispatch that outlives aclose() must not resurrect the
    record file: the row is dropped (logged), the file stays closed, and no
    orphan handle is created."""
    import logging

    out = tmp_path / "run.jsonl"
    recorder = Recorder(_EchoClient(), out)
    routes = recorder.abridge_routes()
    await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "before"}))
    await recorder.aclose()

    with caplog.at_level(logging.WARNING, logger="agentix.bridge.recorder"):
        resp = await routes["/v1/messages"](Request(path="/v1/messages", body={"msg": "late"}))
    assert json.loads(resp.body)["echo"] == "late"  # still served
    assert any("recorder is closed" in r.message for r in caplog.records)
    assert [r["request"]["msg"] for r in _lines(out)] == ["before"]  # no late row
    assert recorder._file is not None and recorder._file.closed  # noqa: SLF001 - not reopened
