"""Unit tests for the Gateway backend pool routing (no model in the loop)."""

from __future__ import annotations

import time

import pytest
from agentix.tito.pool import BackendPool

A, B, C = "http://h1:8000", "http://h2:8000", "http://h3:8000"


def test_requires_backends() -> None:
    with pytest.raises(ValueError):
        BackendPool([])


def test_bad_policy() -> None:
    with pytest.raises(ValueError):
        BackendPool([A], policy="nope")


def test_single_backend_always() -> None:
    pool = BackendPool([A])
    assert pool.pick("s1") == A
    assert pool.pick() == A


def test_sticky_pins_session_to_one_backend() -> None:
    pool = BackendPool([A, B, C], policy="sticky")
    first = pool.pick("rollout-1")
    # Same session keeps hitting the same backend across many turns.
    assert all(pool.pick("rollout-1") == first for _ in range(10))


def test_sticky_spreads_distinct_sessions_round_robin() -> None:
    pool = BackendPool([A, B, C], policy="sticky")
    assigned = [pool.pick(f"s{i}") for i in range(3)]
    assert sorted(assigned) == sorted([A, B, C])  # 3 sessions → 3 distinct backends


def test_round_robin_cycles_every_request() -> None:
    pool = BackendPool([A, B], policy="round_robin")
    assert [pool.pick("ignored") for _ in range(4)] == [A, B, A, B]


def test_down_backend_is_skipped() -> None:
    pool = BackendPool([A, B], policy="round_robin")
    pool.report_down(A)
    assert {pool.pick() for _ in range(6)} == {B}
    pool.report_up(A)
    assert A in {pool.pick() for _ in range(6)}


def test_sticky_session_reassigned_when_backend_down() -> None:
    pool = BackendPool([A, B], policy="sticky")
    pinned = pool.pick("r1")
    pool.report_down(pinned)
    reassigned = pool.pick("r1")
    assert reassigned != pinned
    assert reassigned not in pool._down


def test_all_down_falls_back_not_fails() -> None:
    pool = BackendPool([A, B], policy="round_robin")
    pool.report_down(A)
    pool.report_down(B)
    # Better to attempt a (maybe-recovered) backend than fail routing outright.
    assert pool.pick() in (A, B)


def test_down_backend_retried_after_cooldown() -> None:
    """Nothing in the serving path calls `report_up` for a backend that is
    never picked — without a cooldown a downed replica is blacklisted forever.
    After `down_cooldown` the backend must become eligible again (half-open)."""
    pool = BackendPool([A, B], policy="round_robin", down_cooldown=0.05)
    pool.report_down(A)
    assert {pool.pick() for _ in range(4)} == {B}
    time.sleep(0.06)
    assert A in {pool.pick() for _ in range(4)}


def test_sticky_pin_kept_when_cooldown_expires() -> None:
    """Once the pinned backend's cooldown expires it is retryable — the session
    should return to its original pin (prefix cache) rather than stay migrated."""
    pool = BackendPool([A, B], policy="sticky", down_cooldown=0.05)
    pinned = pool.pick("r1")
    pool.report_down(pinned)
    time.sleep(0.06)
    assert pool.pick("r1") == pinned


def test_sticky_pins_are_bounded_lru() -> None:
    """Session pins are only dropped by an explicit DELETE, but the recommended
    rollout flow never deletes (the trajectory must survive for harvest) — the
    pin map must be bounded, evicting the least-recently-used session."""
    pool = BackendPool([A, B], policy="sticky", max_pins=4)
    pins = {f"s{i}": pool.pick(f"s{i}") for i in range(10)}
    assert len(pool._assigned) == 4  # noqa: SLF001
    # the most recently seen sessions keep their pins ...
    for sid in ("s6", "s7", "s8", "s9"):
        assert pool.pick(sid) == pins[sid]
    # ... and the oldest were evicted (repinning is allowed — stickiness is a
    # cache-locality optimization, not a correctness requirement)
    assert "s0" not in pool._assigned  # noqa: SLF001
