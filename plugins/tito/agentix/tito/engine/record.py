"""Per-turn token record persistence — the gateway's durable capture.

The in-process trajectory (`LinearTrajectory`) is the live state machine;
this module is the crash-safe, append-only export of it. When the gateway is
started with a record directory, every committed chat turn appends exactly
one strict-JSON line to ``<record_dir>/<session_id>.jsonl`` and flushes it,
so the file is complete up to the last committed turn even if the process
dies mid-rollout. Closing a session (DELETE, TTL/capacity eviction,
shutdown) appends one final ``tito.session.v1`` metadata line and closes the
file.

This module is the NORMATIVE definition of the record shape — downstream
consumers adapt to it (see the plugin README for the full schema document).

Two line schemas:

``tito.record.v1`` — one per committed turn (flat structure)::

    {"schema_version": "tito.record.v1", "session_id": ...,
     "thread_id": ...,                      # only when x-thread-id was sent
     "request_id": ..., "turn_index": 0, "ts": <unix seconds>,
     "model": ..., "backend_kind": "sglang" | "vllm",
     "sampling": {"temperature": ..., ...},   # whitelist from the request body
     "tokenizer": {"checkpoint": ..., "tokenizer_sha256": ...,
                   "chat_template_sha256": ...},
     "prompt_token_ids": [...],
     "prompt_segments": [{"start": 0, "end": N, "source": ...}, ...],
     "completion_token_ids": [...],
     "completion_logprobs": [...],          # len == len(completion_token_ids)
     "assistant_message": {...}, "finish_reason": ...,
     "prefix_stable": true,
     "render_skew": null | {"equal": ..., "first_divergence": ...}}

``prompt_segments`` sources: ``render`` (a from-scratch chat-template render
— the first turn, or the fallback when the token prefix window was outrun),
``prefix`` (the reused accumulated checkpoint), one segment per appended
role (``tool`` / ``user`` / ``system``), and ``generation_prompt``.

``prefix_stable`` is computed by the sink against the last line it actually
wrote for the session: true iff this turn's prompt token ids extend the
previous RECORDED turn's ``prompt_token_ids + completion_token_ids``. This
is deliberately not the in-memory checkpoint — a rollback applied for a turn
that never produced a line (upstream non-200, timeout, 409) must surface as
``false`` on the next recorded turn, or a trainer following the contract
would splice a broken token stream.

``turn_index`` is monotonic per session file and advances even when a line
fails to persist, so any dropped line leaves a detectable index gap.

``render_skew`` (vLLM backend only) is the cheap per-turn probe comparing
the render endpoint's from-scratch token ids against the gateway's
accumulated prompt ids; non-equal skew is expected mid-conversation (render
re-renders the echoed history) and is recorded, never enforced.

Strictness: lines are strict JSON (``allow_nan=False``); token ids must be
Python ints and logprobs finite floats — a violating record is NOT written
(logged + index gap), never silently coerced.

``tito.session.v1`` — one final line when the session closes::

    {"schema_version": "tito.session.v1", "session_id": ..., "turns": N,
     "reason": "deleted" | "ttl_evicted" | "capacity_evicted" | "shutdown",
     "ts": <unix seconds>}
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

RECORD_SCHEMA_VERSION = "tito.record.v1"
SESSION_META_SCHEMA_VERSION = "tito.session.v1"

# Request-body keys lifted verbatim into the record's `sampling` object.
# Whitelist, not passthrough: the body also carries messages/tools/stream and
# the fields the gateway itself forces (logprobs et al.), none of which are
# sampling parameters.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "stop",
    "n",
)


def sampling_from_request(request_body: dict[str, Any]) -> dict[str, Any]:
    """The whitelisted sampling parameters present in the chat request."""
    return {key: request_body[key] for key in SAMPLING_KEYS if request_body.get(key) is not None}


def compute_render_skew(
    render_token_ids: list[int] | None, prompt_token_ids: list[int]
) -> dict[str, Any] | None:
    """Compare the backend render's from-scratch prompt ids against the
    gateway's accumulated prompt ids. ``None`` when the backend kind exposes
    no render ids (sglang)."""
    if render_token_ids is None:
        return None
    if render_token_ids == prompt_token_ids:
        return {"equal": True, "first_divergence": None}
    first = next(
        (i for i, (a, b) in enumerate(zip(render_token_ids, prompt_token_ids, strict=False)) if a != b),
        min(len(render_token_ids), len(prompt_token_ids)),
    )
    return {"equal": False, "first_divergence": first}


def build_turn_record(
    *,
    session_id: str,
    request_id: str | None,
    thread_id: str | None,
    model: str | None,
    backend_kind: str,
    sampling: dict[str, Any],
    prompt_token_ids: list[int],
    prompt_segments: list[dict[str, Any]],
    completion_token_ids: list[int],
    completion_logprobs: list[float],
    assistant_message: dict[str, Any],
    finish_reason: str | None,
    tokenizer: dict[str, str],
    render_token_ids: list[int] | None,
) -> dict[str, Any]:
    """One ``tito.record.v1`` line, without the fields the sink assigns
    (``turn_index``, ``prefix_stable``). Pure assembly — validation happens
    in the sink so every rejected record leaves the same detectable gap."""
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "session_id": session_id,
    }
    if thread_id is not None:
        record["thread_id"] = thread_id
    record.update(
        {
            "request_id": request_id,
            "ts": time.time(),
            "model": model,
            "backend_kind": backend_kind,
            "sampling": dict(sampling),
            "tokenizer": dict(tokenizer),
            "prompt_token_ids": list(prompt_token_ids),
            "prompt_segments": list(prompt_segments),
            "completion_token_ids": list(completion_token_ids),
            "completion_logprobs": list(completion_logprobs),
            "assistant_message": assistant_message,
            "finish_reason": finish_reason,
            "render_skew": compute_render_skew(render_token_ids, prompt_token_ids),
        }
    )
    return record


def _validate_turn_record(record: dict[str, Any]) -> None:
    """Strictness gate for token truth: ids are Python ints, logprobs are
    finite floats paired 1:1 with the completion ids. Anything else (numpy
    scalars repr'd to strings, NaN logprobs a backend passed through) must be
    rejected, not silently coerced into training data."""
    for field in ("prompt_token_ids", "completion_token_ids"):
        ids = record[field]
        if not all(type(t) is int for t in ids):
            raise ValueError(f"{field} must be Python ints; got {[type(t).__name__ for t in ids[:5]]}")
    logprobs = record["completion_logprobs"]
    if len(logprobs) != len(record["completion_token_ids"]):
        raise ValueError(
            f"len(completion_logprobs)={len(logprobs)} != "
            f"len(completion_token_ids)={len(record['completion_token_ids'])}"
        )
    if not all(isinstance(lp, float) and math.isfinite(lp) for lp in logprobs):
        raise ValueError("completion_logprobs must be finite floats")


class TurnRecordSink:
    """Per-session JSONL files under ``record_dir``, appended and flushed one
    line per committed turn.

    The sink assigns ``turn_index`` (monotonic per session, advancing even
    when a line fails so gaps are detectable) and computes ``prefix_stable``
    against the last line it actually wrote. Files open lazily on the first
    turn and are closed by ``finalize`` (which also appends the
    ``tito.session.v1`` metadata line). ``finalize`` is idempotent;
    ``close_all`` finalizes every open file (process shutdown). A failed or
    rejected write is logged and never fails the live request — the
    in-memory trajectory still holds the turn for a read-time harvest.
    """

    def __init__(self, record_dir: str | Path) -> None:
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, IO[str]] = {}
        self._turns: dict[str, int] = {}
        # prompt+completion ids of the last successfully written line —
        # the baseline `prefix_stable` is defined against.
        self._last_recorded: dict[str, list[int]] = {}

    def path_for(self, session_id: str) -> Path:
        return self.record_dir / f"{session_id}.jsonl"

    def append_turn(self, session_id: str, record: dict[str, Any]) -> int:
        """Validate and append one turn record (assigning ``turn_index`` and
        ``prefix_stable``), flush, and return the assigned index. Any failure
        is logged and leaves an index gap instead of breaking the turn."""
        turn_index = self._turns.get(session_id, 0)
        try:
            _validate_turn_record(record)
            last = self._last_recorded.get(session_id)
            prompt_ids = record["prompt_token_ids"]
            record = {
                **record,
                "turn_index": turn_index,
                "prefix_stable": not last or prompt_ids[: len(last)] == last,
            }
            self._write(session_id, record)
            self._last_recorded[session_id] = list(prompt_ids) + list(record["completion_token_ids"])
        except Exception:
            # Capture must never take down the serving path: the turn is
            # already committed in memory and remains harvestable via GET.
            # The finally-increment below leaves a detectable turn_index gap.
            logger.exception(
                "tito record: turn %d for session %s NOT persisted (%s)",
                turn_index,
                session_id,
                self.path_for(session_id),
            )
        finally:
            self._turns[session_id] = turn_index + 1
        return turn_index

    def finalize(self, session_id: str, *, reason: str) -> None:
        """Append the final session metadata line and close the file.
        No-op for sessions that never recorded a turn; idempotent."""
        if session_id not in self._files and session_id not in self._turns:
            return
        meta = {
            "schema_version": SESSION_META_SCHEMA_VERSION,
            "session_id": session_id,
            "turns": self._turns.get(session_id, 0),
            "reason": reason,
            "ts": time.time(),
        }
        try:
            self._write(session_id, meta)
        except Exception:
            logger.exception("tito record: failed to finalize %s", self.path_for(session_id))
        finally:
            self._turns.pop(session_id, None)
            self._last_recorded.pop(session_id, None)
            handle = self._files.pop(session_id, None)
            if handle is not None and not handle.closed:
                try:
                    handle.close()
                except OSError:
                    logger.exception("tito record: failed to close %s", self.path_for(session_id))

    def close_all(self, *, reason: str = "shutdown") -> None:
        for session_id in list(self._files):
            self.finalize(session_id, reason=reason)

    def _write(self, session_id: str, record: dict[str, Any]) -> None:
        """Append one strict-JSON line and flush. Raises on any failure —
        the callers own the log-and-continue policy and the gap semantics."""
        handle = self._files.get(session_id)
        if handle is None or handle.closed:
            handle = self.path_for(session_id).open("a", encoding="utf-8")
            self._files[session_id] = handle
        # Strict JSON: no NaN/Infinity literals (allow_nan=False) — a
        # non-finite value slipping past validation fails the line (leaving a
        # gap) rather than emitting non-standard JSON. `default=repr` only
        # covers non-token metadata (e.g. an exotic value inside a sampling
        # field); token ids and logprobs are strictly validated above.
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=repr) + "\n")
        handle.flush()


__all__ = [
    "RECORD_SCHEMA_VERSION",
    "SAMPLING_KEYS",
    "SESSION_META_SCHEMA_VERSION",
    "TurnRecordSink",
    "build_turn_record",
    "compute_render_skew",
    "sampling_from_request",
]
