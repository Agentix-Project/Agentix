"""In-process session + record stores for the gateway.

Persistence (jsonl, parquet, kafka, postgres, an HF dataset, ...) is
deliberately not in core; downstream consumers wrap `SessionStore` /
`RecordStore` or feed `completion_writer.CompletionWriter` instead.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from agentix.gateway.session import Session, SessionResult, SessionStatus


class SessionStore:
    """Thread-safe in-memory registry of live + terminated sessions.

    Live sessions are queryable while running so a coordinator can
    poll status; terminal sessions stick around until evicted, so the
    coordinator can also pull `SessionResult`s asynchronously.
    """

    def __init__(self, *, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._sessions: dict[str, Session] = {}
        self._order: list[str] = []  # insertion order, for eviction
        self._lock = threading.Lock()

    def register(self, session: Session) -> None:
        with self._lock:
            if session.session_id in self._sessions:
                return
            if len(self._sessions) >= self._capacity:
                evicted = self._order.pop(0)
                self._sessions.pop(evicted, None)
            self._sessions[session.session_id] = session
            self._order.append(session.session_id)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_live(self) -> list[Session]:
        with self._lock:
            return [s for s in self._sessions.values() if not s.status.terminal]

    def list_results(self) -> list[SessionResult]:
        with self._lock:
            return [
                s.to_result() for s in self._sessions.values() if s.status.terminal
            ]

    def __iter__(self) -> Iterator[Session]:
        with self._lock:
            return iter(list(self._sessions.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {s.value: 0 for s in SessionStatus}
            for sess in self._sessions.values():
                counts[sess.status.value] += 1
            return {"size": len(self._sessions), "capacity": self._capacity, "by_status": counts}


class RecordStore:
    """Bounded ring of captured LLM-call dicts (per gateway node).

    The gateway funnels every `completion_record` event from every
    session into one node-wide store so a coordinator can pull the
    entire LLM-call log for a run without joining per-session files.
    Per-session records also live on the `Session.records` list.
    """

    def __init__(self, *, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._records: list[dict[str, Any]] = []
        self._dropped = 0
        self._lock = threading.Lock()

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            if len(self._records) >= self._capacity:
                self._records.pop(0)
                self._dropped += 1
            self._records.append(record)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                r for r in self._records if r.get("session_id") == session_id
            ]

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._records),
                "capacity": self._capacity,
                "dropped": self._dropped,
            }


__all__ = ["RecordStore", "SessionStore"]
