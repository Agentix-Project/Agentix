"""In-process completion-record storage for captured LLM calls.

Mirrors `polar.gateway.storage` / `polar.gateway.completion_writer`:
every successful or failed LLM call resolves into one
`CompletionRecord` that the user-facing API can flush to disk,
post to an OTel collector, or attach to a trajectory.

A `CompletionRecord` is captured **once per request**, after the
transform layer has run, so the record always carries:

  * `family`        — what the agent harness asked for.
  * `request_path`  — original URL path the agent hit.
  * `request_body`  — the agent's original (untransformed) JSON.
  * `upstream_body` — the body actually sent to the OpenAI-compatible
                     upstream.
  * `response_body` — the body returned to the agent (in the agent's
                     original API family).
  * `usage`         — extracted token counts when available.

Storage is in-process — the host opens an `InMemoryStore()` for each
proxy session and reads records out before tearing the session down.
Persistence (jsonl, parquet, an external KV) belongs to a downstream
consumer; the bridge keeps zero filesystem state of its own.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from .detection import ApiFamily


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class CompletionRecord:
    """One captured LLM call.

    Identifier is the proxy-side `request_id` so a record can be
    correlated with traces / logs that share the same id.
    """

    request_id: str
    family: ApiFamily
    started_at: float
    ended_at: float
    request_path: str
    request_body: dict[str, Any]
    upstream_body: dict[str, Any]
    response_body: dict[str, Any] | None
    status: str = "ok"
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    # Rollout grouping key, stamped by the bridge so a session's LLM
    # calls form one trajectory. None when the proxy ran un-bridged.
    session_id: str | None = None

    @property
    def duration_ms(self) -> float:
        return max(0.0, (self.ended_at - self.started_at) * 1000.0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["family"] = self.family.value
        return d


class InMemoryStore:
    """Thread-safe bounded in-memory buffer of `CompletionRecord`s.

    Bounded so a long-running agent can't OOM the host. `capacity`
    counts records, not bytes; when the buffer is full, the oldest
    record is dropped and `dropped` is incremented (consumers can read
    `stats()` to detect loss).
    """

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._records: list[CompletionRecord] = []
        self._dropped = 0
        self._lock = threading.Lock()

    def add(self, record: CompletionRecord) -> None:
        with self._lock:
            if len(self._records) >= self._capacity:
                self._records.pop(0)
                self._dropped += 1
            self._records.append(record)

    def snapshot(self) -> list[CompletionRecord]:
        with self._lock:
            return list(self._records)

    def trajectory(self, session_id: str) -> list[CompletionRecord]:
        """The agent-eye LLM-call trajectory for one rollout, in order.

        This is the *text-level* view (request/response bodies) used for
        observability; the token-level trajectory (ids + logprobs) lives
        in the gateway, joined by the same `session_id`.
        """
        with self._lock:
            return [r for r in self._records if r.session_id == session_id]

    def sessions(self) -> list[str]:
        """Distinct session ids seen, in first-appearance order."""
        with self._lock:
            seen: dict[str, None] = {}
            for r in self._records:
                if r.session_id is not None:
                    seen.setdefault(r.session_id, None)
            return list(seen)

    def __iter__(self) -> Iterator[CompletionRecord]:
        return iter(self.snapshot())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._dropped = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._records),
                "capacity": self._capacity,
                "dropped": self._dropped,
            }


def extract_usage(response_body: dict[str, Any] | None, *, family: ApiFamily) -> TokenUsage:
    """Pull token usage out of a response body, family-aware.

    The bridge always stores the *agent's-eye* response body, so we
    look at it in the family the agent expected.
    """
    if not isinstance(response_body, dict):
        return TokenUsage()
    if family.is_anthropic:
        u = response_body.get("usage") or {}
        if not isinstance(u, dict):
            return TokenUsage()
        prompt = int(u.get("input_tokens") or 0)
        completion = int(u.get("output_tokens") or 0)
        cache_read = int(u.get("cache_read_input_tokens") or 0)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cache_read,
            total_tokens=prompt + completion,
        )
    u = response_body.get("usage") or {}
    if not isinstance(u, dict):
        return TokenUsage()
    prompt = int(u.get("prompt_tokens") or 0)
    completion = int(u.get("completion_tokens") or 0)
    total = int(u.get("total_tokens") or 0) or (prompt + completion)
    details = u.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
    c_details = u.get("completion_tokens_details") or {}
    reasoning = (
        int(c_details.get("reasoning_tokens") or 0) if isinstance(c_details, dict) else 0
    )
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        total_tokens=total,
    )


def make_record(
    *,
    request_id: str,
    family: ApiFamily,
    started_at: float,
    request_path: str,
    request_body: dict[str, Any],
    upstream_body: dict[str, Any],
    response_body: dict[str, Any] | None,
    status: str = "ok",
    error: str | None = None,
    ended_at: float | None = None,
    session_id: str | None = None,
) -> CompletionRecord:
    """Factory that fills usage from the response body."""
    record = CompletionRecord(
        request_id=request_id,
        family=family,
        started_at=started_at,
        ended_at=ended_at if ended_at is not None else time.time(),
        request_path=request_path,
        request_body=request_body,
        upstream_body=upstream_body,
        response_body=response_body,
        status=status,
        error=error,
        session_id=session_id,
    )
    record.usage = extract_usage(response_body, family=family)
    return record


__all__ = [
    "CompletionRecord",
    "InMemoryStore",
    "TokenUsage",
    "extract_usage",
    "make_record",
]
