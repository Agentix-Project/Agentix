"""HTTP-surface tests: drive the REAL gateway FastAPI app over ASGI.

Unlike the unit tests (which call `do_proxy` / pool methods directly), these
exercise the full route stack — request parsing, the session lock phases, the
forget-on-delete middleware, error handlers — via real HTTP requests through
`httpx.ASGITransport`. The upstream replica is an `httpx.MockTransport`
swapped into the pooled backend: an OpenAI-shaped chat-completions endpoint
with the `meta_info.output_token_logprobs` the TITO flow requires, keyed by
host so multi-backend routing is observable. The tokenizer is the same tiny
in-memory WordLevel one the engine tests use (no model download).
"""

from __future__ import annotations

import json
import types

import httpx
import pytest
from agentix.tito.pool import BackendPool
from agentix.tito.server import SessionServer
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

A, B = "http://replica-a:8000", "http://replica-b:8000"


@pytest.fixture(scope="module")
def tok():
    specials = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]
    words = ["system", "user", "assistant", "tool", "You", "are", "ok", "done", "Hello"]
    vocab = {t: i for i, t in enumerate(specials + words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    t = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    t.chat_template = (
        "{%- for m in messages -%}<|im_start|>{{ m['role'] }} {{ m['content'] or '' }}<|im_end|>{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|im_start|>assistant {%- endif -%}"
    )
    return t


def _args():
    return types.SimpleNamespace(
        hf_checkpoint="tiny-in-memory",
        chat_template_path=None,
        tito_allowed_append_roles=None,
        tito_model="default",
        session_server_instance_id=None,
        router_timeout=5.0,
    )


class _Replica:
    """OpenAI-shaped fake replica behind an httpx.MockTransport. Records every
    body it sees; hosts in `down` raise a transport error instead of answering.
    `message` / `usage` / `raw_json` allow per-test response shaping."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.down: set[str] = set()
        self.message: dict = {"role": "assistant", "content": "ok done"}
        self.usage: dict = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        self.raw_json: dict | None = None  # full-body override (still 200)
        self.raw_content: bytes | None = None  # verbatim body override (still 200)

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in self.down:
            raise httpx.ConnectError(f"{host} refused")
        if request.method == "DELETE":
            return httpx.Response(204)
        body = json.loads(request.content)
        self.calls.append((host, body))
        if self.raw_content is not None:
            return httpx.Response(
                200, content=self.raw_content, headers={"content-type": "application/json"}
            )
        if self.raw_json is not None:
            return httpx.Response(200, json=self.raw_json)
        completion_ids = [7, 8]  # "ok done" in the tiny vocab
        return httpx.Response(
            200,
            json={
                "id": "c1", "object": "chat.completion", "model": "m",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": self.message,
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, t, ""] for t in completion_ids],
                        "completion_tokens": len(completion_ids),
                    },
                }],
                "usage": self.usage,
            },
        )


@pytest.fixture()
def server(tok, monkeypatch):
    """(SessionServer over the tiny tokenizer, replica, pool) with a 2-replica pool."""
    monkeypatch.setattr(
        "agentix.tito.engine.session_app.load_tokenizer", lambda *a, **k: tok
    )
    pool = BackendPool([A, B], policy="sticky", down_cooldown=0.05)
    srv = SessionServer(_args(), pool)
    replica = _Replica()
    srv._backend.client = httpx.AsyncClient(
        transport=httpx.MockTransport(replica.handler), timeout=5.0
    )
    return srv, replica, pool


@pytest.fixture()
def gateway(server):
    """(http client for the real app, replica, pool)."""
    srv, replica, pool = server
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://gw", timeout=5.0
    )
    return client, replica, pool


_CHAT = {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}


@pytest.mark.asyncio
async def test_full_session_flow_over_http(gateway):
    client, replica, _ = gateway
    r = await client.post("/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok done"
    # the gateway injected pretokenized input_ids + forced logprobs upstream
    _, seen = replica.calls[0]
    assert seen["logprobs"] is True
    assert isinstance(seen["input_ids"], list) and seen["input_ids"]

    r = await client.get(f"/sessions/{sid}")
    assert r.status_code == 200
    got = r.json()
    assert len(got["records"]) == 1
    assert got["metadata"]["accumulated_token_ids"][-2:] == [7, 8]

    r = await client.delete(f"/sessions/{sid}")
    assert r.status_code == 204
    r = await client.get(f"/sessions/{sid}")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_stream_request_is_forced_non_streaming(gateway):
    """The TITO flow needs the full JSON completion (logprobs + meta_info), so
    the gateway must force stream=false upstream and answer 200 with the JSON
    body — not 500 on an unparseable SSE stream."""
    client, replica, _ = gateway
    sid = (await client.post("/sessions")).json()["session_id"]

    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions", json={**_CHAT, "stream": True}
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok done"
    _, seen = replica.calls[0]
    assert seen["stream"] is False


@pytest.mark.asyncio
async def test_tool_call_completion_with_null_content_is_accepted(gateway):
    """Tool-call-only assistant turns routinely carry content:null (sglang's
    parser consumes all text; vLLM emits None) — the gateway must record the
    turn, not 502 an agentic rollout at its first tool call."""
    client, replica, _ = gateway
    replica.message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "compute", "arguments": "{}"},
        }],
    }
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    got = (await client.get(f"/sessions/{sid}")).json()
    assert len(got["records"]) == 1  # the turn was recorded


@pytest.mark.asyncio
async def test_content_none_without_tool_calls_is_still_502(gateway):
    client, replica, _ = gateway
    replica.message = {"role": "assistant", "content": None}
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 502


@pytest.mark.asyncio
async def test_malformed_request_body_is_400(gateway):
    client, _, _ = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_malformed_200_upstream_is_clean_502(gateway):
    """A 200 upstream body without the expected structure (choices/meta_info
    shapes) must surface as a clean 502, not an unhandled 500."""
    client, replica, _ = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    for weird in ({"weird": True}, {"choices": []}, {"choices": [{"meta_info": None}]}):
        replica.raw_json = weird
        r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
        assert r.status_code == 502, weird


@pytest.mark.asyncio
async def test_upstream_nan_json_passes_through(gateway):
    """The proxy must not re-encode the upstream JSON body — NaN/Infinity are
    valid for Python's json but not for a strict re-serializer, and re-encoding
    also perturbs key order/whitespace for byte-sensitive consumers."""
    client, replica, _ = gateway
    # stdlib json accepts NaN both ways; httpx's own json= encoder does not,
    # so the replica ships the body verbatim.
    replica.raw_content = json.dumps({
        "id": "c1", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok done"},
            "meta_info": {
                "output_token_logprobs": [[float("nan"), 7, ""], [-0.1, 8, ""]],
                "completion_tokens": 2,
            },
        }],
        "usage": {},
    }).encode()
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert b"NaN" in r.content


@pytest.mark.asyncio
async def test_registry_tokenizer_load_is_not_trust_remote_code(tok, monkeypatch):
    """trust_remote_code executes checkpoint-shipped Python at startup — it
    must be an explicit opt-in, never the hardcoded default."""
    seen: dict = {}

    def fake_load(name, chat_template_path=None, trust_remote_code=False):
        seen["trust_remote_code"] = trust_remote_code
        return tok

    monkeypatch.setattr("agentix.tito.engine.session_app.load_tokenizer", fake_load)
    pool = BackendPool([A])
    SessionServer(_args(), pool)
    assert seen["trust_remote_code"] is False


@pytest.mark.asyncio
async def test_unknown_session_is_404_over_http(gateway):
    client, _, _ = gateway
    r = await client.post("/sessions/nope/v1/chat/completions", json=_CHAT)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_backend_failover_and_recovery_over_http(gateway):
    """A replica that refuses connections is marked down and the session is
    repinned to the survivor; once the cooldown passes the failed replica is
    eligible again for new sessions."""
    import asyncio

    client, replica, pool = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    pinned = pool.pick(sid)

    replica.down.add(httpx.URL(pinned).host)
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 502  # transport error surfaces as 502, marks down

    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200  # repinned to the healthy replica
    survivor = replica.calls[-1][0]
    assert survivor != httpx.URL(pinned).host

    replica.down.clear()
    await asyncio.sleep(0.06)  # let the down cooldown expire
    hosts = {pool.pick(f"fresh-{i}") for i in range(4)}
    assert pinned in hosts  # recovered replica takes new sessions again


@pytest.mark.asyncio
async def test_delete_forgets_pin_over_http(gateway):
    client, _, pool = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert sid in pool._assigned  # noqa: SLF001
    r = await client.delete(f"/sessions/{sid}")
    assert r.status_code == 204
    assert sid not in pool._assigned  # noqa: SLF001


@pytest.mark.asyncio
async def test_proxied_subresource_delete_keeps_pin(gateway):
    """Only `DELETE /sessions/{sid}` ends the session; a DELETE proxied to a
    backend sub-resource under the session prefix must not drop the sticky pin."""
    client, _, pool = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert sid in pool._assigned  # noqa: SLF001
    r = await client.delete(f"/sessions/{sid}/v1/some/backend/resource")
    assert r.status_code == 204
    assert sid in pool._assigned  # noqa: SLF001


@pytest.mark.asyncio
async def test_slow_request_body_does_not_hold_session_lock(server):
    """The session lock must not be held while reading the request body — a
    dribbling client upload would otherwise pin the lock indefinitely, wedging
    DELETE and every other operation on the session."""
    import asyncio

    srv, _, _ = server
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://gw", timeout=5.0
    )
    sid = (await client.post("/sessions")).json()["session_id"]
    session = srv.app.state.tito_registry.get_session(sid)

    payload = json.dumps(_CHAT).encode()
    started = asyncio.Event()
    release = asyncio.Event()

    async def dribble():
        yield payload[:5]
        started.set()
        await release.wait()
        yield payload[5:]

    task = asyncio.create_task(client.post(
        f"/sessions/{sid}/v1/chat/completions",
        content=dribble(),
        headers={"content-type": "application/json"},
    ))
    await started.wait()
    await asyncio.sleep(0.01)  # let the handler park inside the body read
    locked_during_upload = session.lock.locked()
    release.set()
    r = await task
    assert r.status_code == 200
    assert locked_during_upload is False


@pytest.mark.asyncio
async def test_cancelled_delete_does_not_brick_session(server):
    """A DELETE cancelled while waiting for the session lock (client timeout /
    disconnect) must leave the session deletable — not wedged behind a stuck
    closing flag with the trajectory leaked in the registry."""
    import asyncio
    import contextlib

    srv, _, _ = server
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://gw", timeout=5.0
    )
    sid = (await client.post("/sessions")).json()["session_id"]
    session = srv.app.state.tito_registry.get_session(sid)

    await session.lock.acquire()  # simulate another request holding the lock
    try:
        del_task = asyncio.create_task(client.delete(f"/sessions/{sid}"))
        await asyncio.sleep(0.05)  # let the DELETE park at lock acquisition
        del_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
            await del_task
    finally:
        session.lock.release()

    r = await client.delete(f"/sessions/{sid}")
    assert r.status_code == 204
