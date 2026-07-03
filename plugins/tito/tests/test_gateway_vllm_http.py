"""HTTP-surface tests for the vLLM backend kind: drive the REAL gateway FastAPI
app over ASGI against a fake vLLM replica.

vLLM's chat-completions endpoint silently ignores an sglang-style ``input_ids``
field, so the vLLM turn is a three-call chain against the token-native API
(vLLM >= 0.24.0): ``/v1/chat/completions/render`` translates the agent's chat
request into a ``GenerateRequest`` (its from-scratch ``token_ids`` are
discarded), ``/inference/v1/generate`` runs on the gateway's exact pretokenized
prompt ids, and ``/v1/chat/completions/derender`` turns the raw token output
back into a chat completion (running the server-side tool/reasoning parsers).
The fake replica below implements those three endpoints with the v0.24.0 wire
shapes — including the released ``GenerateResponse`` carrying no ``usage``.
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

A = "http://replica-a:8000"


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
        backend_kind="vllm",
        chat_template_path=None,
        tito_allowed_append_roles=None,
        tito_model="default",
        session_server_instance_id=None,
        router_timeout=5.0,
    )


_RENDER = "/v1/chat/completions/render"
_GENERATE = "/inference/v1/generate"
_DERENDER = "/v1/chat/completions/derender"


class _VllmReplica:
    """Fake vLLM replica speaking the v0.24.0 render/generate/derender wire
    shapes. Records every body it sees per endpoint; `fail` forces an error
    status per endpoint; `generate_raw` / `derender_raw_content` override the
    200 bodies for malformed-upstream tests."""

    def __init__(self) -> None:
        self.calls: dict[str, list[dict]] = {"render": [], "generate": [], "derender": []}
        self.fail: dict[str, tuple[int, dict]] = {}
        self.rendered_ids = [99, 98]  # from-scratch render — the gateway must NOT generate from these
        self.sampling_params = {"temperature": 0.5, "max_tokens": 32}
        self.completion_ids = [7, 8]
        self.finish_reason = "stop"
        self.message: dict = {"role": "assistant", "content": "ok done"}
        self.generate_raw: dict | None = None
        self.derender_raw_content: bytes | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        name = {_RENDER: "render", _GENERATE: "generate", _DERENDER: "derender"}.get(request.url.path)
        if name is None:
            return httpx.Response(404, json={"error": {"message": f"no route {request.url.path}"}})
        body = json.loads(request.content)
        self.calls[name].append(body)
        if name in self.fail:
            status, err = self.fail[name]
            return httpx.Response(status, json=err)
        return getattr(self, f"_{name}")(body)

    def _render(self, body: dict) -> httpx.Response:
        # Deliberately leak `stream: true` + stream_options the way a real
        # render would if the chat request streamed (vLLM copies the flag) —
        # the gateway must overwrite both before calling generate.
        return httpx.Response(200, json={
            "request_id": "chatcmpl-render-1",
            "token_ids": list(self.rendered_ids),
            "sampling_params": dict(self.sampling_params),
            "model": body.get("model"),
            "stream": True,
            "stream_options": {"include_usage": True},
        })

    def _generate(self, body: dict) -> httpx.Response:
        if self.generate_raw is not None:
            return httpx.Response(200, json=self.generate_raw)
        ids = list(self.completion_ids)
        return httpx.Response(200, json={
            # v0.24.0 shape: no usage/model/created, random request_id.
            "request_id": "9f0e6d1c",
            "choices": [{
                "index": 0,
                "finish_reason": self.finish_reason,
                "token_ids": ids,
                "logprobs": {"content": [
                    {"token": f"token_id:{t}", "logprob": -0.1, "bytes": None, "top_logprobs": []} for t in ids
                ]},
            }],
            "prompt_logprobs": None,
        })

    def _derender(self, body: dict) -> httpx.Response:
        if self.derender_raw_content is not None:
            return httpx.Response(
                200, content=self.derender_raw_content, headers={"content-type": "application/json"}
            )
        gen = body["generate_response"]
        prompt_tokens = body.get("prompt_tokens") or 0
        completion_tokens = sum(len(c.get("token_ids") or []) for c in gen.get("choices", []))
        return httpx.Response(200, json={
            "id": gen.get("request_id", "x"), "object": "chat.completion", "created": 1,
            "model": body["model"],
            "choices": [{
                "index": c.get("index", 0),
                # real derender passes finish_reason through verbatim — it
                # never rewrites to "tool_calls"; the gateway does that.
                "finish_reason": c.get("finish_reason"),
                "message": dict(self.message),
                "stop_reason": None,
            } for c in gen.get("choices", [])],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })


@pytest.fixture()
def server(tok, monkeypatch):
    monkeypatch.setattr(
        "agentix.tito.engine.session_app.load_tokenizer", lambda *a, **k: tok
    )
    pool = BackendPool([A])
    srv = SessionServer(_args(), pool)
    replica = _VllmReplica()
    srv._backend.client = httpx.AsyncClient(
        transport=httpx.MockTransport(replica.handler), timeout=5.0
    )
    return srv, replica


@pytest.fixture()
def gateway(server):
    srv, replica = server
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://gw", timeout=5.0
    )
    return client, replica


_CHAT = {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}


@pytest.mark.asyncio
async def test_full_vllm_session_flow_over_http(gateway):
    """Token-in: generate runs on the gateway's pretokenized ids, not render's.
    Token-out: the accumulated trajectory is exactly prompt + generated ids."""
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]

    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok done"

    prompt_ids = replica.calls["generate"][0]["token_ids"]
    assert prompt_ids and prompt_ids != replica.rendered_ids

    got = (await client.get(f"/sessions/{sid}")).json()
    assert len(got["records"]) == 1
    assert got["metadata"]["accumulated_token_ids"] == prompt_ids + [7, 8]


@pytest.mark.asyncio
async def test_vllm_second_turn_reuses_token_prefix(gateway):
    """The derendered assistant message echoed back with an appended tool turn
    must extend the stored history (no spurious rollback) and reuse the stored
    token prefix."""
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    turn1_accumulated = (await client.get(f"/sessions/{sid}")).json()["metadata"]["accumulated_token_ids"]

    followup = {
        "model": "m",
        "messages": [
            *_CHAT["messages"],
            {"role": "assistant", "content": "ok done"},
            {"role": "tool", "content": "done"},
        ],
    }
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=followup)
    assert r.status_code == 200

    turn2_prompt_ids = replica.calls["generate"][1]["token_ids"]
    assert turn2_prompt_ids[: len(turn1_accumulated)] == turn1_accumulated
    got = (await client.get(f"/sessions/{sid}")).json()
    assert len(got["records"]) == 2
    assert got["metadata"]["accumulated_token_ids"] == turn2_prompt_ids + [7, 8]


@pytest.mark.asyncio
async def test_vllm_forces_token_recording_fields(gateway):
    """The chat body sent to render is forced non-streaming with logprobs on
    and carries no sglang input_ids; the GenerateRequest is re-forced
    non-streaming (render leaks stream) with the prompt ids substituted and
    render's resolved sampling params preserved."""
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]

    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={**_CHAT, "stream": True, "stream_options": {"include_usage": True}},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok done"

    render_body = replica.calls["render"][0]
    assert render_body["logprobs"] is True
    assert render_body["top_logprobs"] == 0
    assert render_body["stream"] is False
    assert "stream_options" not in render_body
    assert "input_ids" not in render_body

    generate_body = replica.calls["generate"][0]
    assert generate_body["stream"] is False
    assert "stream_options" not in generate_body
    assert generate_body["sampling_params"]["temperature"] == 0.5
    assert generate_body["sampling_params"]["max_tokens"] == 32


@pytest.mark.asyncio
async def test_vllm_derender_request_carries_context(gateway):
    """Derender needs the model (parser lookup), the exact prompt token count
    (usage), the verbatim generate response, and the original chat request
    (tool/reasoning parsers read tools + tool_choice from it)."""
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)

    derender_body = replica.calls["derender"][0]
    assert derender_body["model"] == "m"
    assert derender_body["prompt_tokens"] == len(replica.calls["generate"][0]["token_ids"])
    assert derender_body["chat_request"]["messages"] == _CHAT["messages"]
    assert derender_body["generate_response"]["choices"][0]["token_ids"] == [7, 8]


@pytest.mark.asyncio
async def test_vllm_tool_call_turn_rewrites_finish_reason(gateway):
    """derender passes finish_reason through verbatim ("stop"), unlike vLLM's
    own chat endpoint — the gateway rewrites it to "tool_calls" so agent loops
    that branch on finish_reason behave identically on both backends."""
    client, replica = gateway
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
    choice = r.json()["choices"][0]
    assert choice["message"]["tool_calls"][0]["id"] == "call_1"
    assert choice["finish_reason"] == "tool_calls"
    got = (await client.get(f"/sessions/{sid}")).json()
    assert len(got["records"]) == 1
    assert got["records"][0]["response"]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_vllm_untouched_derender_body_passes_through_verbatim(gateway):
    """When no finish_reason fixup applies, the derendered bytes reach the
    agent verbatim — no re-encode perturbing whitespace or key order."""
    client, replica = gateway
    replica.derender_raw_content = (
        b'{"id":"v1","object":"chat.completion","model":"m",'
        b'"choices":[{"index":0,"finish_reason":"stop",'
        b'"message":{"role":"assistant","content":"ok done"}}],"usage":{}}'
    )
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    assert r.content == replica.derender_raw_content


@pytest.mark.asyncio
async def test_vllm_content_none_without_tool_calls_is_502(gateway):
    client, replica = gateway
    replica.message = {"role": "assistant", "content": None}
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 502


@pytest.mark.asyncio
async def test_vllm_missing_model_is_400(gateway):
    """derender requires `model`; fail fast before any upstream call."""
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert r.status_code == 400
    assert replica.calls["render"] == []


@pytest.mark.asyncio
async def test_vllm_render_error_passes_through(gateway):
    client, replica = gateway
    replica.fail["render"] = (400, {"error": {"message": "bad chat request", "type": "BadRequestError", "code": 400}})
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 400
    assert r.json()["error"]["message"] == "bad chat request"
    assert replica.calls["generate"] == []
    assert (await client.get(f"/sessions/{sid}")).json()["records"] == []


@pytest.mark.asyncio
async def test_vllm_generate_error_passes_through(gateway):
    client, replica = gateway
    replica.fail["generate"] = (500, {"error": {"message": "engine dead", "type": "InternalServerError", "code": 500}})
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 500
    assert replica.calls["derender"] == []
    assert (await client.get(f"/sessions/{sid}")).json()["records"] == []


@pytest.mark.asyncio
async def test_vllm_derender_error_passes_through(gateway):
    client, replica = gateway
    replica.fail["derender"] = (
        503, {"error": {"message": "parser overloaded", "type": "ServiceUnavailableError", "code": 503}}
    )
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 503
    assert (await client.get(f"/sessions/{sid}")).json()["records"] == []


@pytest.mark.asyncio
async def test_vllm_malformed_generate_body_is_502(gateway):
    """A 200 generate body without a sound token_ids list is the backend's
    fault — clean 502, never an unhandled 500, and derender is never called."""
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    for weird in (
        {"weird": True},
        {"choices": []},
        {"choices": [{"index": 0, "finish_reason": "stop"}]},          # token_ids missing
        {"choices": [{"index": 0, "token_ids": []}]},                  # empty
        {"choices": [{"index": 0, "token_ids": ["a", "b"]}]},          # non-int
    ):
        replica.generate_raw = weird
        r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
        assert r.status_code == 502, weird
    assert replica.calls["derender"] == []


@pytest.mark.asyncio
async def test_vllm_logprobs_count_mismatch_is_502(gateway):
    """The gateway forces logprobs on; a generate response whose logprobs
    entries don't pair 1:1 with token_ids (or lack logprobs entirely) signals
    the forcing was ignored or the body is corrupt — the same cross-check the
    sglang path runs against meta_info.completion_tokens."""
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    for weird in (
        {"choices": [{"index": 0, "token_ids": [7, 8]}]},  # logprobs missing
        {"choices": [{"index": 0, "token_ids": [7, 8], "logprobs": {"content": [
            {"token": "token_id:7", "logprob": -0.1, "bytes": None, "top_logprobs": []},
        ]}}]},
    ):
        replica.generate_raw = weird
        r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
        assert r.status_code == 502, weird


@pytest.mark.asyncio
async def test_vllm_malformed_derender_body_is_502(gateway):
    client, replica = gateway
    sid = (await client.post("/sessions")).json()["session_id"]
    for weird in (b"not json", b'{"choices": []}', b'{"choices": [{"index": 0}]}'):
        replica.derender_raw_content = weird
        r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
        assert r.status_code == 502, weird


def test_unknown_backend_kind_fails_at_startup(tok, monkeypatch):
    monkeypatch.setattr("agentix.tito.engine.session_app.load_tokenizer", lambda *a, **k: tok)
    args = _args()
    args.backend_kind = "tgi"
    with pytest.raises(ValueError, match="backend_kind"):
        SessionServer(args, BackendPool([A]))
