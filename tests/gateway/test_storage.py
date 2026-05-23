"""Tests for the gateway's in-memory session + record stores."""

from __future__ import annotations

from agentix.gateway.session import Session, SessionSpec, SessionStatus
from agentix.gateway.storage import RecordStore, SessionStore


def _spec() -> SessionSpec:
    return SessionSpec(callable_ref="x::y", image="i", bundle="b")


def test_session_store_register_and_get() -> None:
    store = SessionStore()
    sess = Session(spec=_spec())
    store.register(sess)
    assert store.get(sess.session_id) is sess
    assert store.get("nope") is None
    assert len(store) == 1


def test_session_store_evicts_oldest_at_capacity() -> None:
    store = SessionStore(capacity=2)
    a = Session(spec=_spec())
    b = Session(spec=_spec())
    c = Session(spec=_spec())
    store.register(a)
    store.register(b)
    store.register(c)
    assert store.get(a.session_id) is None
    assert store.get(b.session_id) is b
    assert store.get(c.session_id) is c


def test_session_store_partitions_live_vs_results() -> None:
    store = SessionStore()
    live = Session(spec=_spec())
    done = Session(spec=_spec())
    done.mark(SessionStatus.SUCCEEDED)
    store.register(live)
    store.register(done)
    assert store.list_live() == [live]
    results = store.list_results()
    assert len(results) == 1
    assert results[0].status is SessionStatus.SUCCEEDED


def test_record_store_drops_oldest_at_capacity() -> None:
    store = RecordStore(capacity=2)
    for i in range(4):
        store.add({"session_id": "s", "i": i})
    snap = store.snapshot()
    assert [r["i"] for r in snap] == [2, 3]
    stats = store.stats()
    assert stats == {"size": 2, "capacity": 2, "dropped": 2}


def test_record_store_per_session_filter() -> None:
    store = RecordStore()
    store.add({"session_id": "a", "i": 1})
    store.add({"session_id": "b", "i": 2})
    store.add({"session_id": "a", "i": 3})
    assert [r["i"] for r in store.for_session("a")] == [1, 3]
    assert [r["i"] for r in store.for_session("b")] == [2]
