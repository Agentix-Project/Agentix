"""Contract tests for per-turn record persistence and session lifecycle.

Drive the REAL gateway app over ASGI (same harness as test_gateway_http /
test_gateway_vllm_http: tiny in-memory WordLevel tokenizer, fake replica via
httpx.MockTransport) and assert the durable-capture contract:

- one flushed `tito.record.v1` JSON line per committed turn (crash-safe:
  the file is complete after every turn, before any close);
- the record shape: ids/logprobs 1:1, prompt segment spans, tokenizer
  fingerprint, request_id from the x-request-id header;
- `prefix_stable` true on the linear path, false (but still served +
  recorded) after a history rewrite that rolls back a checkpoint;
- interleaved turns on one session are an explicit 409, never a silently
  dropped update;
- TTL / capacity eviction finalizes the record file first and never touches
  an in-flight session; DELETE appends the final `tito.session.v1` line;
- no --record-dir -> no files, behavior unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import types
from pathlib import Path

import httpx
import pytest
from agentix.tito.engine.record import compute_render_skew
from agentix.tito.pool import BackendPool
from agentix.tito.server import SessionServer
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

A = "http://replica-a:8000"

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}


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


def _args(**overrides):
    ns = types.SimpleNamespace(
        hf_checkpoint="tiny-in-memory",
        chat_template_path=None,
        tito_allowed_append_roles=None,
        tito_model="default",
        session_server_instance_id=None,
        router_timeout=5.0,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


class _Replica:
    """sglang-shaped fake replica; optionally parks a request on an event
    (for the interleave test)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.completion_ids = [7, 8]  # "ok done"
        self.logprobs = [-0.25, -0.5]
        self.message: dict = {"role": "assistant", "content": "ok done"}
        self.hold: asyncio.Event | None = None
        self.hold_marker: str | None = None
        self.fail_next = False
        self.delay = 0.0

    async def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        if self.hold is not None and self.hold_marker in json.dumps(body):
            await self.hold.wait()
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_next:
            self.fail_next = False
            return httpx.Response(503, json={"error": "upstream busy"})
        ids = list(self.completion_ids)
        # Serialize with stdlib json (allows NaN like a lenient real backend
        # would) and ship verbatim — httpx's own `json=` encoder is strict.
        blob = json.dumps({
            "id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": dict(self.message),
                "meta_info": {
                    "output_token_logprobs": [[lp, t, ""] for lp, t in zip(self.logprobs, ids, strict=True)],
                    "completion_tokens": len(ids),
                },
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": len(ids), "total_tokens": 5},
        }).encode()
        return httpx.Response(200, content=blob, headers={"content-type": "application/json"})


def _make_gateway(tok, monkeypatch, **arg_overrides):
    monkeypatch.setattr("agentix.tito.engine.session_app.load_tokenizer", lambda *a, **k: tok)
    pool = BackendPool([A])
    srv = SessionServer(_args(**arg_overrides), pool)
    replica = _Replica()
    srv._backend.client = httpx.AsyncClient(transport=httpx.MockTransport(replica.handler), timeout=5.0)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=srv.app), base_url="http://gw", timeout=5.0)
    return client, replica, srv, pool


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _records(path: Path) -> list[dict]:
    return [r for r in _lines(path) if r["schema_version"] == "tito.record.v1"]


def _meta(path: Path) -> list[dict]:
    return [r for r in _lines(path) if r["schema_version"] == "tito.session.v1"]


@pytest.mark.asyncio
async def test_record_line_shape_and_crash_safety(tok, monkeypatch, tmp_path):
    """One complete, flushed record line per committed turn — readable after
    EVERY turn with the file still open (a crash mid-rollout loses nothing
    already committed) — carrying the full tito.record.v1 shape."""
    client, replica, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path))
    sid = (await client.post("/sessions")).json()["session_id"]
    path = tmp_path / f"{sid}.jsonl"

    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={**_CHAT, "temperature": 0.6, "top_p": 0.95, "max_tokens": 128},
        headers={"x-request-id": "req-001", "x-thread-id": "thread-7"},
    )
    assert r.status_code == 200

    # Crash safety: the line is on disk NOW, without any close/flush call.
    (rec,) = _records(path)
    accumulated = (await client.get(f"/sessions/{sid}")).json()["metadata"]["accumulated_token_ids"]

    assert rec["schema_version"] == "tito.record.v1"
    assert rec["session_id"] == sid
    assert rec["thread_id"] == "thread-7"  # x-thread-id passthrough
    assert rec["turn_index"] == 0
    assert rec["request_id"] == "req-001"
    assert rec["model"] == "m"
    assert rec["backend_kind"] == "sglang"
    # Whitelisted sampling params lifted verbatim from the request body.
    assert rec["sampling"] == {"temperature": 0.6, "top_p": 0.95, "max_tokens": 128}
    assert rec["prompt_token_ids"] + rec["completion_token_ids"] == accumulated
    assert rec["completion_token_ids"] == [7, 8]
    assert rec["completion_logprobs"] == [-0.25, -0.5]
    assert len(rec["completion_logprobs"]) == len(rec["completion_token_ids"])
    assert rec["assistant_message"]["content"] == "ok done"
    assert rec["finish_reason"] == "stop"
    assert rec["prefix_stable"] is True
    assert rec["render_skew"] is None  # sglang exposes no render ids
    assert rec["tokenizer"] == {
        "checkpoint": "tiny-in-memory",
        "tokenizer_sha256": hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
    }
    # First turn: one from-scratch render segment covering the whole prompt.
    assert rec["prompt_segments"] == [
        {"start": 0, "end": len(rec["prompt_token_ids"]), "source": "render"}
    ]
    assert "ts" in rec

    # Turn 2 (tool append): line 2 is on disk immediately; segments show the
    # reused prefix boundary and the per-role suffixes.
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
    rec1, rec2 = _records(path)
    assert rec2["turn_index"] == 1
    assert rec2["request_id"] is None  # no x-request-id header sent
    assert "thread_id" not in rec2  # key omitted when x-thread-id absent
    assert rec2["sampling"] == {}  # no sampling params in the request
    assert rec2["prefix_stable"] is True
    assert [s["source"] for s in rec2["prompt_segments"]] == ["prefix", "tool", "generation_prompt"]
    # Segment spans tile the prompt exactly, and the prefix span IS the
    # previous checkpoint (prompt + completion of turn 1).
    assert rec2["prompt_segments"][0]["start"] == 0
    prefix_end = rec2["prompt_segments"][0]["end"]
    assert rec2["prompt_token_ids"][:prefix_end] == rec1["prompt_token_ids"] + rec1["completion_token_ids"]
    for left, right in zip(rec2["prompt_segments"], rec2["prompt_segments"][1:], strict=False):
        assert left["end"] == right["start"]
    assert rec2["prompt_segments"][-1]["end"] == len(rec2["prompt_token_ids"])


@pytest.mark.asyncio
async def test_history_rewrite_records_prefix_stable_false_and_still_serves(tok, monkeypatch, tmp_path):
    """A compaction-style rewrite of the last tool turn rolls back one
    checkpoint: the turn is SERVED and RECORDED, flagged prefix_stable=false
    so a trainer can split or reject it instead of splicing a lie."""
    client, replica, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path))
    sid = (await client.post("/sessions")).json()["session_id"]
    path = tmp_path / f"{sid}.jsonl"

    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    base = [*_CHAT["messages"], {"role": "assistant", "content": "ok done"}]
    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": "m", "messages": [*base, {"role": "tool", "content": "done"}]},
    )
    assert r.status_code == 200

    # Rewrite the tool turn (divergent content) — single-step rollback.
    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": "m", "messages": [*base, {"role": "tool", "content": "You"}]},
    )
    assert r.status_code == 200

    recs = _records(path)
    assert [rec["prefix_stable"] for rec in recs] == [True, True, False]
    assert [rec["turn_index"] for rec in recs] == [0, 1, 2]
    # The rewritten turn still tiles cleanly over its own prompt.
    assert [s["source"] for s in recs[2]["prompt_segments"]] == ["prefix", "tool", "generation_prompt"]


@pytest.mark.asyncio
async def test_interleaved_turn_is_explicit_409(tok, monkeypatch, tmp_path):
    """Two in-flight turns on one session: the loser's completion cannot be
    committed to the rewound/advanced trajectory. It must be an explicit 409
    — the previous behavior served the response and silently dropped it from
    the capture, which is unacceptable data loss in production."""
    client, replica, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path))
    sid = (await client.post("/sessions")).json()["session_id"]

    replica.hold = asyncio.Event()
    replica.hold_marker = "HOLD-ME"

    slow = asyncio.create_task(client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "Hello HOLD-ME"}]},
    ))
    while not replica.calls:  # let the slow turn reach the backend
        await asyncio.sleep(0.01)

    fast = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert fast.status_code == 200  # the interleaver wins and is recorded

    replica.hold.set()
    lost = await slow
    assert lost.status_code == 409
    assert "changed while the turn was in flight" in lost.json()["error"]

    # Exactly the winner is recorded — no phantom line for the 409.
    recs = _records(tmp_path / f"{sid}.jsonl")
    assert len(recs) == 1
    assert recs[0]["prompt_token_ids"]  # the fast turn's record


@pytest.mark.asyncio
async def test_delete_finalizes_record_file(tok, monkeypatch, tmp_path):
    """DELETE-after-harvest: the documented rollout end appends the final
    tito.session.v1 line and closes the file."""
    client, _, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path))
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert (await client.delete(f"/sessions/{sid}")).status_code == 204

    path = tmp_path / f"{sid}.jsonl"
    (meta,) = _meta(path)
    assert meta == {
        "schema_version": "tito.session.v1",
        "session_id": sid,
        "turns": 1,
        "reason": "deleted",
        "ts": meta["ts"],
    }
    assert _lines(path)[-1] == meta  # the meta line is the LAST line


@pytest.mark.asyncio
async def test_ttl_eviction_flushes_record_file_and_forgets_pin(tok, monkeypatch, tmp_path):
    client, _, srv, pool = _make_gateway(
        tok, monkeypatch, record_dir=str(tmp_path), session_ttl_seconds=0.05
    )
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert sid in pool._assigned  # noqa: SLF001

    await asyncio.sleep(0.06)
    registry = srv.app.state.tito_registry
    assert registry.sweep() == [sid]

    path = tmp_path / f"{sid}.jsonl"
    (meta,) = _meta(path)
    assert meta["reason"] == "ttl_evicted"
    assert meta["turns"] == 1
    assert len(_records(path)) == 1  # committed turns survived the eviction
    assert sid not in pool._assigned  # noqa: SLF001 - eviction drops the sticky pin
    assert (await client.get(f"/sessions/{sid}")).status_code == 404


@pytest.mark.asyncio
async def test_eviction_never_touches_inflight_sessions(tok, monkeypatch, tmp_path):
    """A session with a turn parked on the backend (lock NOT held — phase 2)
    must survive TTL and capacity sweeps until the turn finishes."""
    client, replica, srv, _ = _make_gateway(
        tok, monkeypatch, record_dir=str(tmp_path), session_ttl_seconds=0.01, max_sessions=1
    )
    sid = (await client.post("/sessions")).json()["session_id"]

    replica.hold = asyncio.Event()
    replica.hold_marker = "HOLD-ME"
    slow = asyncio.create_task(client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "Hello HOLD-ME"}]},
    ))
    while not replica.calls:
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.02)  # idle past the TTL while in flight
    registry = srv.app.state.tito_registry
    assert registry.sweep() == []  # in-flight: skipped by TTL AND capacity
    assert sid in registry.sessions

    replica.hold.set()
    assert (await slow).status_code == 200
    await asyncio.sleep(0.02)
    assert registry.sweep() == [sid]  # idle again -> now evictable


@pytest.mark.asyncio
async def test_capacity_eviction_is_lru_and_flushes(tok, monkeypatch, tmp_path):
    client, _, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path), max_sessions=1)
    sid_old = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid_old}/v1/chat/completions", json=_CHAT)
    sid_new = (await client.post("/sessions")).json()["session_id"]

    # The next request's sweep trims the overflow: LRU (sid_old) goes.
    r = await client.post(f"/sessions/{sid_new}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200

    registry = srv.app.state.tito_registry
    assert sid_old not in registry.sessions
    assert sid_new in registry.sessions
    (meta,) = _meta(tmp_path / f"{sid_old}.jsonl")
    assert meta["reason"] == "capacity_evicted"


@pytest.mark.asyncio
async def test_no_record_dir_writes_nothing_and_behavior_is_unchanged(tok, monkeypatch, tmp_path):
    client, _, srv, _ = _make_gateway(tok, monkeypatch)  # no record_dir
    sid = (await client.post("/sessions")).json()["session_id"]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200
    got = (await client.get(f"/sessions/{sid}")).json()
    assert len(got["records"]) == 1
    assert srv.app.state.tito_registry.record_sink is None
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere


@pytest.mark.asyncio
async def test_shutdown_finalizes_open_record_files(tok, monkeypatch, tmp_path):
    client, _, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path))
    sid = (await client.post("/sessions")).json()["session_id"]
    await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)

    registry = srv.app.state.tito_registry
    assert registry.close in srv.app.router.on_shutdown  # lifespan wiring
    registry.close()
    (meta,) = _meta(tmp_path / f"{sid}.jsonl")
    assert meta["reason"] == "shutdown"


@pytest.mark.asyncio
async def test_unrecorded_rollback_turn_still_breaks_prefix_stability(tok, monkeypatch, tmp_path):
    """Adversarial-review regression: a rollback applied in phase 1 whose
    turn never produced a record line (upstream 503) must NOT vanish from the
    record stream — the next successful line's prefix_stable is computed
    against the last RECORDED line, not the in-memory checkpoint, so the
    discontinuity is flagged."""
    client, replica, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path))
    sid = (await client.post("/sessions")).json()["session_id"]
    path = tmp_path / f"{sid}.jsonl"

    assert (await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)).status_code == 200
    base = [*_CHAT["messages"], {"role": "assistant", "content": "ok done"}]
    assert (await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": "m", "messages": [*base, {"role": "tool", "content": "done"}]},
    )).status_code == 200

    # History rewrite (rollback applied in phase 1), upstream 503s -> passed
    # through, NO record line, but the rollback stays committed in memory.
    replica.fail_next = True
    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": "m", "messages": [*base, {"role": "tool", "content": "You"}]},
    )
    assert r.status_code == 503

    # The agent retries the same rewritten history; the in-memory checkpoint
    # now extends cleanly — but the record stream does not.
    r = await client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": "m", "messages": [*base, {"role": "tool", "content": "You"}]},
    )
    assert r.status_code == 200

    recs = _records(path)
    assert len(recs) == 3  # the 503 turn left no line (and no index gap: it never committed)
    prev_stream = recs[1]["prompt_token_ids"] + recs[1]["completion_token_ids"]
    assert recs[2]["prompt_token_ids"][: len(prev_stream)] != prev_stream
    assert [rec["prefix_stable"] for rec in recs] == [True, True, False]


@pytest.mark.asyncio
async def test_generation_time_does_not_count_as_idle_for_ttl(tok, monkeypatch, tmp_path):
    """Adversarial-review regression: the idle clock starts when a turn ENDS.
    A single generation slower than the TTL must not let the agent's
    immediately following request sweep its own live session away."""
    client, replica, srv, _ = _make_gateway(
        tok, monkeypatch, record_dir=str(tmp_path), session_ttl_seconds=0.5
    )
    sid = (await client.post("/sessions")).json()["session_id"]

    replica.delay = 0.8  # one turn's generation exceeds the TTL
    assert (await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)).status_code == 200
    replica.delay = 0.0

    followup = {
        "model": "m",
        "messages": [
            *_CHAT["messages"],
            {"role": "assistant", "content": "ok done"},
            {"role": "tool", "content": "done"},
        ],
    }
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=followup)
    assert r.status_code == 200  # the session survived: zero actual idle time
    assert len(_records(tmp_path / f"{sid}.jsonl")) == 2


@pytest.mark.asyncio
async def test_invalid_token_data_is_rejected_with_detectable_gap(tok, monkeypatch, tmp_path):
    """Strict-JSON sink contract: a NaN logprob (JSON-parseable upstream, not
    strict JSON) or a non-int token id must never be coerced into the record
    file — the line is dropped, the turn still serves, and the next line's
    turn_index shows a detectable gap."""
    client, replica, srv, _ = _make_gateway(tok, monkeypatch, record_dir=str(tmp_path))
    sid = (await client.post("/sessions")).json()["session_id"]
    path = tmp_path / f"{sid}.jsonl"

    # Turn 0: NaN logprob passed through by the backend (stdlib json accepts it).
    replica.logprobs = [float("nan"), -0.5]
    r = await client.post(f"/sessions/{sid}/v1/chat/completions", json=_CHAT)
    assert r.status_code == 200  # capture failure never fails the served turn
    assert not path.exists() or _records(path) == []

    # Turn 1: clean — recorded with a turn_index gap exposing the dropped line.
    replica.logprobs = [-0.25, -0.5]
    followup = {
        "model": "m",
        "messages": [
            *_CHAT["messages"],
            {"role": "assistant", "content": "ok done"},
            {"role": "tool", "content": "done"},
        ],
    }
    assert (await client.post(f"/sessions/{sid}/v1/chat/completions", json=followup)).status_code == 200
    (rec,) = _records(path)
    assert rec["turn_index"] == 1  # index 0 is the detectable hole
    # The dropped line is also the stability baseline hole: turn 1 extends
    # nothing recorded, so it is the stream's first line and reads stable.
    assert rec["prefix_stable"] is True
    # Every persisted line is strict JSON (no NaN/Infinity literals).
    for line in path.read_text().splitlines():
        json.loads(line, parse_constant=lambda name: pytest.fail(f"non-strict JSON literal {name}"))
    # The session meta line still counts every turn slot, recorded or not.
    assert (await client.delete(f"/sessions/{sid}")).status_code == 204
    (meta,) = _meta(path)
    assert meta["turns"] == 2


def test_compute_render_skew_contract():
    assert compute_render_skew(None, [1, 2]) is None
    assert compute_render_skew([1, 2], [1, 2]) == {"equal": True, "first_divergence": None}
    assert compute_render_skew([9, 2], [1, 2]) == {"equal": False, "first_divergence": 0}
    assert compute_render_skew([1, 2], [1, 2, 3]) == {"equal": False, "first_divergence": 2}
    assert compute_render_skew([1, 2, 3], [1, 2]) == {"equal": False, "first_divergence": 2}
