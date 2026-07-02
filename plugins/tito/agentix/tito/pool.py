"""Backend pool — route OpenAI-compatible requests across N base URLs.

The TITO Gateway accepts one *or more* OpenAI-compatible backend URLs
(sglang replicas — the token-recording chat flow needs sglang's ``meta_info``
extension; the raw proxy routes work with any OpenAI-compatible server) and
forwards each request to one of them. This is the routing layer, independent
of TITO tokenization, so it is unit-tested on its own with no model in the
loop.

Policy:
  - ``sticky`` (default): each ``session_id`` is pinned to one backend,
    chosen round-robin among healthy backends on first sight and then
    remembered. A multi-turn rollout reuses one replica's prefix KV-cache,
    which is the right default for TITO. (TITO sends explicit ``input_ids``,
    so any replica *can* serve any turn — stickiness is a cache-locality
    optimization, not a correctness requirement.)
  - ``round_robin``: spread every request across healthy backends.

Backends reported down via ``report_down`` are skipped; a sticky session
whose backend goes down is reassigned on its next pick. Recovery is
two-pronged: the proxy layer calls ``report_up`` whenever a backend answers
a request, and a down mark expires after ``down_cooldown`` seconds anyway
(half-open: the backend becomes eligible again; if it is still broken the
next failed request re-marks it). Sticky pins are LRU-bounded at
``max_pins`` — the recommended rollout flow never DELETEs its session (the
trajectory must survive for harvest), so an unbounded pin map would grow
forever. Losing an old pin only costs prefix-cache locality, never
correctness.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence

_POLICIES = ("sticky", "round_robin")


class BackendPool:
    def __init__(
        self,
        backends: Sequence[str],
        *,
        policy: str = "sticky",
        down_cooldown: float = 30.0,
        max_pins: int = 10_000,
    ) -> None:
        urls = [b.rstrip("/") for b in backends if b]
        if not urls:
            raise ValueError("BackendPool requires at least one backend url")
        if policy not in _POLICIES:
            raise ValueError(f"policy must be one of {_POLICIES}; got {policy!r}")
        if max_pins < 1:
            raise ValueError(f"max_pins must be >= 1; got {max_pins}")
        self._backends = urls
        self._policy = policy
        self._rr = 0
        self._assigned: OrderedDict[str, str] = OrderedDict()
        self._down: dict[str, float] = {}  # url -> monotonic time marked down
        self._down_cooldown = down_cooldown
        self._max_pins = max_pins
        self._lock = threading.Lock()

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(self._backends)

    def _is_down(self, backend: str, now: float) -> bool:
        marked = self._down.get(backend)
        return marked is not None and (now - marked) < self._down_cooldown

    def _healthy(self, now: float) -> list[str]:
        healthy = [b for b in self._backends if not self._is_down(b, now)]
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
            now = time.monotonic()
            healthy = self._healthy(now)
            if self._policy == "sticky" and session_id is not None:
                current = self._assigned.get(session_id)
                if current is not None and not self._is_down(current, now):
                    self._assigned.move_to_end(session_id)
                    return current
                chosen = self._next_round_robin(healthy)
                self._assigned[session_id] = chosen
                self._assigned.move_to_end(session_id)
                while len(self._assigned) > self._max_pins:
                    self._assigned.popitem(last=False)
                return chosen
            return self._next_round_robin(healthy)

    def report_down(self, backend: str) -> None:
        with self._lock:
            self._down[backend.rstrip("/")] = time.monotonic()

    def report_up(self, backend: str) -> None:
        with self._lock:
            self._down.pop(backend.rstrip("/"), None)

    def forget(self, session_id: str) -> None:
        """Drop a session's sticky assignment (call when the rollout ends)."""
        with self._lock:
            self._assigned.pop(session_id, None)


__all__ = ["BackendPool"]
