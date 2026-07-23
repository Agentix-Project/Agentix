"""Per-turn token record persistence — the gateway's durable capture.

The in-process trajectory (`LinearTrajectory`) is the live state machine;
this module is the crash-safe, append-only export of it. When the gateway is
started with a record directory, every committed chat turn appends exactly
one JSON line to ``<record_dir>/<session_id>.jsonl`` and flushes it, so the
file is complete up to the last committed turn even if the process dies
mid-rollout. Closing a session (DELETE, TTL/capacity eviction, shutdown)
appends one final ``tito.session.v1`` metadata line and closes the file.

Two line schemas:

``tito.record.v1`` — one per committed turn::

    {"schema_version": "tito.record.v1", "session_id": ..., "turn_index": 0,
     "request_id": ..., "model": ..., "backend_kind": "sglang" | "vllm",
     "prompt_token_ids": [...], "completion_token_ids": [...],
     "completion_logprobs": [...],          # len == len(completion_token_ids)
     "prompt_segments": [{"start": 0, "end": N, "source": ...}, ...],
     "assistant_message": {...}, "finish_reason": ...,
     "tokenizer_fingerprint": {"checkpoint": ..., "chat_template_sha256": ...},
     "prefix_stable": true, "render_skew": null | {"equal": ..., "first_divergence": ...},
     "ts": <unix seconds>}

``prompt_segments`` sources: ``render`` (a from-scratch chat-template render
— the first turn, or the fallback when the token prefix window was outrun),
``prefix`` (the reused accumulated checkpoint), one segment per appended
role (``tool`` / ``user`` / ``system``), and ``generation_prompt``.

``prefix_stable`` is true iff this turn's prompt token ids extend the last
*committed* checkpoint (previous prompt + completion). A retry rollback or a
history rewrite records ``false`` — the turn is still served and recorded,
but a trainer must not splice it into one linear token stream.

``render_skew`` (vLLM backend only) is the cheap per-turn probe comparing
the render endpoint's from-scratch token ids against the gateway's
accumulated prompt ids; non-equal skew is expected mid-conversation (render
re-renders the echoed history) and is recorded, never enforced.

``tito.session.v1`` — one final line when the session closes::

    {"schema_version": "tito.session.v1", "session_id": ..., "turns": N,
     "reason": "deleted" | "ttl_evicted" | "capacity_evicted" | "shutdown",
     "ts": <unix seconds>}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import IO, Any

logger = logging.getLogger(__name__)

RECORD_SCHEMA_VERSION = "tito.record.v1"
SESSION_META_SCHEMA_VERSION = "tito.session.v1"


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
    model: str | None,
    backend_kind: str,
    prompt_token_ids: list[int],
    prompt_segments: list[dict[str, Any]],
    prefix_stable: bool,
    completion_token_ids: list[int],
    completion_logprobs: list[float],
    assistant_message: dict[str, Any],
    finish_reason: str | None,
    tokenizer_fingerprint: dict[str, str],
    render_token_ids: list[int] | None,
) -> dict[str, Any]:
    """One JSON-serializable ``tito.record.v1`` line (without ``turn_index``,
    which the sink assigns monotonically per session file)."""
    if len(completion_logprobs) != len(completion_token_ids):
        raise ValueError(
            f"len(completion_logprobs)={len(completion_logprobs)} != "
            f"len(completion_token_ids)={len(completion_token_ids)}"
        )
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "session_id": session_id,
        "request_id": request_id,
        "model": model,
        "backend_kind": backend_kind,
        "prompt_token_ids": list(prompt_token_ids),
        "completion_token_ids": list(completion_token_ids),
        "completion_logprobs": list(completion_logprobs),
        "prompt_segments": list(prompt_segments),
        "assistant_message": assistant_message,
        "finish_reason": finish_reason,
        "tokenizer_fingerprint": dict(tokenizer_fingerprint),
        "prefix_stable": bool(prefix_stable),
        "render_skew": compute_render_skew(render_token_ids, prompt_token_ids),
        "ts": time.time(),
    }


class TurnRecordSink:
    """Per-session JSONL files under ``record_dir``, appended and flushed one
    line per committed turn.

    Files open lazily on the first turn and are closed by ``finalize`` (which
    also appends the ``tito.session.v1`` metadata line). ``finalize`` is
    idempotent; ``close_all`` finalizes every open file (process shutdown).
    A failed disk write is logged and never fails the live request — the
    in-memory trajectory still holds the turn for a read-time harvest.
    """

    def __init__(self, record_dir: str | Path) -> None:
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, IO[str]] = {}
        self._turns: dict[str, int] = {}

    def path_for(self, session_id: str) -> Path:
        return self.record_dir / f"{session_id}.jsonl"

    def append_turn(self, session_id: str, record: dict[str, Any]) -> int:
        """Append one turn record (assigning ``turn_index``), flush, and
        return the assigned index."""
        turn_index = self._turns.get(session_id, 0)
        record = {**record, "turn_index": turn_index}
        self._write(session_id, record)
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
        finally:
            self._turns.pop(session_id, None)
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
        try:
            handle = self._files.get(session_id)
            if handle is None or handle.closed:
                handle = self.path_for(session_id).open("a", encoding="utf-8")
                self._files[session_id] = handle
            # `default=repr` keeps a stray non-JSON value (and `allow_nan`
            # keeps a NaN logprob the backend passed through) from sinking
            # the line — best-effort capture beats failing the live turn.
            handle.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")
            handle.flush()
        except OSError:
            # Capture must never take down the serving path: the turn is
            # already committed in memory and remains harvestable via GET.
            logger.exception(
                "tito record: failed to append to %s — turn NOT persisted", self.path_for(session_id)
            )


__all__ = [
    "RECORD_SCHEMA_VERSION",
    "SESSION_META_SCHEMA_VERSION",
    "TurnRecordSink",
    "build_turn_record",
    "compute_render_skew",
]
