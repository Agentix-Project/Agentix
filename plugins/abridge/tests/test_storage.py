"""Tests for `agentix.bridge.storage`."""

from __future__ import annotations

import time

from agentix.bridge.detection import ApiFamily
from agentix.bridge.storage import (
    CompletionRecord,
    InMemoryStore,
    TokenUsage,
    extract_usage,
    make_record,
)


def test_extract_usage_openai_with_details() -> None:
    body = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 7},
        }
    }
    usage = extract_usage(body, family=ApiFamily.OPENAI_CHAT_COMPLETIONS)
    assert usage == TokenUsage(
        prompt_tokens=100,
        completion_tokens=30,
        cached_tokens=20,
        reasoning_tokens=7,
        total_tokens=130,
    )


def test_extract_usage_anthropic_with_cache_read() -> None:
    body = {
        "usage": {
            "input_tokens": 80,
            "output_tokens": 12,
            "cache_read_input_tokens": 15,
        }
    }
    usage = extract_usage(body, family=ApiFamily.ANTHROPIC_MESSAGES)
    assert usage == TokenUsage(
        prompt_tokens=80, completion_tokens=12, cached_tokens=15, total_tokens=92
    )


def test_extract_usage_missing_body() -> None:
    assert extract_usage(None, family=ApiFamily.OPENAI_CHAT_COMPLETIONS) == TokenUsage()


def test_make_record_round_trip() -> None:
    body = {"messages": [], "model": "m"}
    record = make_record(
        request_id="rid-1",
        family=ApiFamily.ANTHROPIC_MESSAGES,
        started_at=1.0,
        ended_at=1.25,
        request_path="/v1/messages",
        request_body=body,
        upstream_body={"model": "openai-m", "messages": []},
        response_body={
            "id": "msg",
            "content": [],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )
    assert record.usage.prompt_tokens == 5
    assert record.usage.completion_tokens == 2
    assert abs(record.duration_ms - 250.0) < 1e-6

    d = record.to_dict()
    assert d["family"] == "anthropic.messages"
    assert d["usage"]["prompt_tokens"] == 5


def test_in_memory_store_capacity_and_drops() -> None:
    store = InMemoryStore(capacity=3)
    for i in range(5):
        store.add(
            CompletionRecord(
                request_id=f"r{i}",
                family=ApiFamily.OPENAI_CHAT_COMPLETIONS,
                started_at=0.0,
                ended_at=0.0,
                request_path="/v1/chat/completions",
                request_body={},
                upstream_body={},
                response_body={},
            )
        )
    snap = store.snapshot()
    assert [r.request_id for r in snap] == ["r2", "r3", "r4"]
    stats = store.stats()
    assert stats == {"size": 3, "capacity": 3, "dropped": 2}


def test_in_memory_store_thread_safety_smoke() -> None:
    import threading

    store = InMemoryStore(capacity=200)

    def writer(start: int) -> None:
        for i in range(50):
            store.add(
                CompletionRecord(
                    request_id=f"r{start + i}",
                    family=ApiFamily.OPENAI_CHAT_COMPLETIONS,
                    started_at=time.time(),
                    ended_at=time.time(),
                    request_path="/v1/chat/completions",
                    request_body={},
                    upstream_body={},
                    response_body={},
                )
            )

    threads = [threading.Thread(target=writer, args=(s * 1000,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store) == 200


def test_store_trajectory_groups_by_session():
    from agentix.bridge import CompletionRecord, InMemoryStore
    from agentix.bridge.detection import ApiFamily

    store = InMemoryStore()

    def rec(rid: str, sid: str | None) -> CompletionRecord:
        return CompletionRecord(
            request_id=rid, family=ApiFamily.OPENAI_CHAT_COMPLETIONS,
            started_at=0.0, ended_at=1.0, request_path="/v1/chat/completions",
            request_body={}, upstream_body={}, response_body={}, session_id=sid,
        )

    for rid, sid in (("a", "s1"), ("b", "s2"), ("c", "s1"), ("d", None)):
        store.add(rec(rid, sid))
    assert [r.request_id for r in store.trajectory("s1")] == ["a", "c"]
    assert store.sessions() == ["s1", "s2"]
