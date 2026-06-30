"""Backend pool — route OpenAI-compatible requests across N base URLs.

The TITO Gateway accepts one *or more* OpenAI-compatible backend URLs
(sglang/vLLM replicas) and forwards each request to one of them. This is
the routing layer, independent of TITO tokenization, so it is unit-tested
on its own with no model in the loop.

Policy:
  - ``sticky`` (default): each ``session_id`` is pinned to one backend,
    chosen round-robin among healthy backends on first sight and then
    remembered. A multi-turn rollout reuses one replica's prefix KV-cache,
    which is the right default for TITO. (TITO sends explicit ``input_ids``,
    so any replica *can* serve any turn — stickiness is a cache-locality
    optimization, not a correctness requirement.)
  - ``round_robin``: spread every request across healthy backends.

Backends reported down via ``report_down`` are skipped until ``report_up``;
a sticky session whose backend goes down is reassigned on its next pick.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

_POLICIES = ("sticky", "round_robin")


class BackendPool:
    def __init__(self, backends: Sequence[str], *, policy: str = "sticky") -> None:
        urls = [b.rstrip("/") for b in backends if b]
        if not urls:
            raise ValueError("BackendPool requires at least one backend url")
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}; got {policy!r}")
        self._backends = urls
        self._policy = policy
        self._rr = 0
        self._assigned: dict[str, str] = {}
        self._down: set[str] = set()
        self._lock = threading.Lock()

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(self._backends)

    def _healthy(self) -> list[str]:
        healthy = [b for b in self._backends if b not in self._down]
        # All down → fall back to the full set rather than fail the request;
        # the forward attempt surfaces the real error.
        return healthy or list(self._backends)

    def _next_round_robin(self, healthy: list[str]) -> str:
        chosen = healthy[self._rr % len(healthy)]
        self._rr += 1
        return chosen

    def pick(self, session_id: str | None = None) -> str:
        """Choose a backend for a request. With the sticky policy and a
        `session_id`, return that session's pinned backend (assigning one
        the first time, or reassigning if the pinned one is down)."""
        with self._lock:
            healthy = self._healthy()
            if self._policy == "sticky" and session_id is not None:
                current = self._assigned.get(session_id)
                if current is not None and current not in self._down:
                    return current
                chosen = self._next_round_robin(healthy)
                self._assigned[session_id] = chosen
                return chosen
            return self._next_round_robin(healthy)

    def report_down(self, backend: str) -> None:
        with self._lock:
            self._down.add(backend.rstrip("/"))

    def report_up(self, backend: str) -> None:
        with self._lock:
            self._down.discard(backend.rstrip("/"))

    def forget(self, session_id: str) -> None:
        """Drop a session's sticky assignment (call when the rollout ends)."""
        with self._lock:
            self._assigned.pop(session_id, None)


__all__ = ["BackendPool"]
