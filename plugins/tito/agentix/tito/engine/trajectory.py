"""Linear trajectory state machine + session registry.

`LinearTrajectory` holds one session's message history and accumulated token-ID
checkpoints, and is the heart of incremental pretokenization: on each turn it
validates that the request extends the stored history (rolling back at most one
assistant step on agent retries) and reuses the stored token prefix. `SessionRegistry`
maps session IDs to trajectories and computes the from-scratch-vs-accumulated
mismatch report. Mutating methods must be called under `LinearTrajectory.lock`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .compare import TokenSeqComparator
from .errors import MessageValidationError, SessionNotFoundError, TokenizationError
from .messages import assert_messages_append_only_with_allowed_role, message_matches
from .pretokenize import TITOTokenizer
from .record import TurnRecordSink

logger = logging.getLogger(__name__)

# Only single-step rollback is supported (an agent retrying one tool call).
MAX_ASSISTANT_ROLLBACK_STEPS = 1


class SessionRecord(BaseModel):
    timestamp: float
    method: str
    path: str
    request: dict
    response: dict
    status_code: int


class GetSessionResponse(BaseModel):
    session_id: str
    records: list[SessionRecord]
    metadata: dict = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _RollbackPlan:
    """A computed-but-not-applied rollback: message truncation point, the
    checkpoint to land on, and how many trailing checkpoints to discard."""

    msg_end: int
    checkpoint_index: int
    discard_count: int


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """The pretokenized prompt for one turn, plus the capture metadata the
    per-turn record needs.

    `segments` are half-open ``{"start", "end", "source"}`` spans over
    `token_ids`; sources are ``render`` (from-scratch chat-template render),
    ``prefix`` (the reused accumulated checkpoint), one per appended role
    (``tool``/``user``/``system``), and ``generation_prompt``.

    `prefix_stable` is True iff `token_ids` extends the last *committed*
    in-memory checkpoint (previous prompt + completion) — False after a retry
    rollback or when the checkpoint window was outrun and the prompt was
    re-rendered. NOTE: this is engine-level, advisory metadata. The
    `prefix_stable` field in a persisted `tito.record.v1` line is computed by
    the record sink against the last RECORDED line instead — an applied
    rollback whose turn never produced a line (upstream error) must still
    surface as a break in the record stream (see `record.TurnRecordSink`).
    """

    token_ids: list[int]
    segments: list[dict[str, Any]]
    prefix_stable: bool


def _spans(parts: list[tuple[str, list[int]]]) -> list[dict[str, Any]]:
    """Turn ``(source, ids)`` parts into cumulative-offset segment spans,
    dropping empty parts (a template may render an empty suffix)."""
    segments: list[dict[str, Any]] = []
    offset = 0
    for source, ids in parts:
        if not ids:
            continue
        segments.append({"start": offset, "end": offset + len(ids), "source": source})
        offset += len(ids)
    return segments


@dataclass
class LinearTrajectory:
    """Message history + accumulated token-ID checkpoints for one session."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    # Requests currently being served on this session (any phase, including
    # the un-locked upstream exchange). The registry never evicts a session
    # whose count is non-zero.
    inflight: int = field(default=0, repr=False, compare=False)
    # Idle clock for TTL eviction; the registry touches it on every lookup.
    last_used: float = field(default_factory=time.monotonic, repr=False, compare=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
    records: list[SessionRecord] = field(default_factory=list)
    trajectory_token_ids: list[list[int]] = field(default_factory=list)
    num_assistant: int = 0
    # Monotonic change counter, bumped by BOTH update and rollback. Unlike
    # `num_assistant` (which rollback decrements and update restores — an ABA
    # hazard), an equal `version` proves the session was untouched in between.
    version: int = 0

    @property
    def token_ids(self) -> list[int]:
        """The latest assistant checkpoint's token IDs."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_record(self, record: SessionRecord) -> None:
        self.records.append(record)

    def prepare_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
    ) -> list[int]:
        """Build the full prompt token IDs for *request_messages*. First turn renders
        from scratch; later turns reuse the stored token prefix (rolling back at most
        one assistant step on a retry). Must be called under ``self.lock``."""
        return self.prepare_prompt(request_messages, tools, tito_tokenizer=tito_tokenizer).token_ids

    def prepare_prompt(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
    ) -> PreparedPrompt:
        """`prepare_pretokenized` plus the capture metadata: segment spans over
        the prompt ids and the prefix-stability verdict against the last
        committed checkpoint. Must be called under ``self.lock``."""
        # The stability baseline is the checkpoint as COMMITTED — captured
        # before any rollback below mutates it.
        committed = list(self.token_ids)

        if not self.messages:
            ids = tito_tokenizer.render_messages(
                request_messages, tools=tools, add_generation_prompt=True, tokenize=True
            )
            return PreparedPrompt(token_ids=ids, segments=_spans([("render", ids)]), prefix_stable=True)

        # Plan first, mutate last: a request that fails validation must be a
        # pure 4xx with NO committed rollback side effects — otherwise a
        # rejected request silently truncates the trajectory and bricks the
        # original branch.
        plan = self._plan_rollback(request_messages)
        base_messages = self.messages if plan is None else self.messages[: plan.msg_end]
        try:
            assert_messages_append_only_with_allowed_role(
                base_messages, request_messages, tito_tokenizer.allowed_append_roles
            )
        except ValueError as e:
            raise MessageValidationError(f"{e}; to allow more roles use --tito-allowed-append-roles") from e
        if plan is not None:
            self._apply_rollback(plan)

        if not self.token_ids:
            # The token prefix is unavailable — either a retry chain walked
            # past the trimmed checkpoint window, or a prior turn never landed
            # its update. Re-render from scratch: incremental == from-scratch
            # is the engine invariant, so this is lossless (never merge onto
            # an empty prefix, which would drop the whole stored history).
            ids = tito_tokenizer.render_messages(
                request_messages, tools=tools, add_generation_prompt=True, tokenize=True
            )
            parts: list[tuple[str, list[int]]] = [("render", ids)]
        else:
            # Same construction as `TITOTokenizer.merge_tokens`, decomposed so
            # the segment boundaries survive into the per-turn record.
            prefix = tito_tokenizer.fix_prefix(self.token_ids)
            appended = tito_tokenizer.appended_segments(self.messages, request_messages, tools)
            ids = prefix + [tid for _, seg_ids in appended for tid in seg_ids]
            parts = [("prefix", prefix), *appended]

        prefix_stable = not committed or ids[: len(committed)] == committed
        return PreparedPrompt(token_ids=ids, segments=_spans(parts), prefix_stable=prefix_stable)

    def update_pretokenized_state(
        self,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        max_trim_tokens: int,
    ) -> None:
        """Append ``prompt+completion`` token IDs as a new checkpoint after a successful
        response, validating the previously-stored IDs are a prefix (tolerating up to
        ``max_trim_tokens`` trailing differences). Must be called under ``self.lock``."""
        all_token_ids = prompt_token_ids + completion_token_ids

        prev = self.token_ids
        if prev:
            check_len = len(prev) - max_trim_tokens
            if check_len > 0 and all_token_ids[:check_len] != prev[:check_len]:
                # strict=False: the new sequence may legitimately be SHORTER
                # than the stored checkpoint (that length difference IS the
                # mismatch) — the default below covers the exhausted tail.
                first_mismatch = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=False))
                        if a != b
                    ),
                    min(len(all_token_ids), check_len),
                )
                raise TokenizationError(
                    f"pretokenized prefix mismatch: stored {len(prev)} tokens "
                    f"(checking first {check_len}, allowing {max_trim_tokens} trailing) are not a prefix of "
                    f"prompt_token_ids + completion_token_ids ({len(all_token_ids)} tokens), "
                    f"first mismatch at index {first_mismatch}, matched {first_mismatch}/{check_len} prefix tokens\n"
                    f"request_messages={request_messages}\nassistant_message={assistant_message}"
                )

        self.messages = list(request_messages) + [assistant_message]
        self.trajectory_token_ids.append(all_token_ids)
        self.num_assistant += 1
        self.version += 1
        # Single-step rollback (MAX_ASSISTANT_ROLLBACK_STEPS=1) can only ever
        # reach the last two checkpoints; keeping every full prefix+completion
        # list would be O(turns^2) dead memory per session. (A retry chain
        # that outruns this window falls back to a from-scratch render in
        # `prepare_pretokenized`.)
        if len(self.trajectory_token_ids) > MAX_ASSISTANT_ROLLBACK_STEPS + 1:
            del self.trajectory_token_ids[: -(MAX_ASSISTANT_ROLLBACK_STEPS + 1)]

    def _plan_rollback(self, request_messages: list[dict[str, Any]]) -> _RollbackPlan | None:
        """If *request_messages* diverges from the stored history, compute the
        rollback to the last assistant checkpoint within the matching prefix
        (single-step only). Pure — mutates nothing; `None` means no divergence."""
        stored = self.messages
        if not stored:
            return None

        match_len = 0
        for i in range(min(len(request_messages), len(stored))):
            if message_matches(stored[i], request_messages[i]):
                match_len = i + 1
            else:
                break

        if match_len >= len(stored):
            return None

        rollback_msg_end = 0
        checkpoint_index = -1
        assistant_count = 0
        for i in range(match_len):
            if stored[i].get("role") == "assistant":
                rollback_msg_end = i + 1
                checkpoint_index = assistant_count
                assistant_count += 1

        if checkpoint_index < 0:
            raise MessageValidationError(
                f"rollback failed: no assistant message found in the first {match_len} matched messages "
                f"(stored has {len(stored)} messages, request has {len(request_messages)} messages)"
            )

        discard_count = self.num_assistant - (checkpoint_index + 1)
        if discard_count > MAX_ASSISTANT_ROLLBACK_STEPS:
            raise MessageValidationError(
                f"rollback failed: discard_count={discard_count} exceeds "
                f"max_assistant_rollback_steps={MAX_ASSISTANT_ROLLBACK_STEPS} "
                f"(stored has {len(stored)} messages, request has {len(request_messages)} messages)"
            )
        return _RollbackPlan(
            msg_end=rollback_msg_end, checkpoint_index=checkpoint_index, discard_count=discard_count
        )

    def _apply_rollback(self, plan: _RollbackPlan) -> None:
        logger.info(
            "Rolling back session: stored %d messages / %d assistants -> checkpoint %d (messages[:%d]), "
            "discarding %d assistant(s)",
            len(self.messages), self.num_assistant, plan.checkpoint_index, plan.msg_end, plan.discard_count,
        )
        self.messages = self.messages[: plan.msg_end]
        # trajectory_token_ids holds only the trailing checkpoints (older ones
        # are trimmed as unreachable), so discard relative to the end — the
        # absolute checkpoint_index no longer maps onto the list. A retry chain
        # can legally outrun the trimmed window: clamp, possibly to empty (the
        # caller then re-renders from scratch instead of merging).
        if plan.discard_count and self.trajectory_token_ids:
            del self.trajectory_token_ids[-min(plan.discard_count, len(self.trajectory_token_ids)):]
        self.records = self.records[: plan.checkpoint_index + 1]
        self.num_assistant = plan.checkpoint_index + 1
        self.version += 1


class SessionRegistry:
    """Session ID -> trajectory map + shared tokenizer/comparator, plus the
    optional durable capture and lifecycle policy for long-running gateways.

    CRUD plus the read-only mismatch computation; never mutates trajectory
    *token* state itself. From ``args`` (all optional, default off):

    - ``record_dir``          — per-session JSONL turn records (`TurnRecordSink`).
    - ``session_ttl_seconds`` — evict sessions idle longer than this.
    - ``max_sessions``        — LRU-evict beyond this many sessions.

    Eviction always flushes + finalizes the session's record file first and
    NEVER touches a session with in-flight requests (``inflight`` > 0, a held
    lock, or a pending close). ``on_evict`` lets the server layer drop
    per-session routing state (e.g. the pool's sticky pin).
    """

    def __init__(self, args: Any, tokenizer: Any, *, tito_tokenizer: TITOTokenizer) -> None:
        self.sessions: OrderedDict[str, LinearTrajectory] = OrderedDict()
        self.args = args
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator: TokenSeqComparator = tito_tokenizer.create_comparator()
        record_dir = getattr(args, "record_dir", None)
        self.record_sink: TurnRecordSink | None = TurnRecordSink(record_dir) if record_dir else None
        self.session_ttl_seconds: float | None = getattr(args, "session_ttl_seconds", None) or None
        self.max_sessions: int | None = getattr(args, "max_sessions", None) or None
        self.on_evict: Callable[[str], None] | None = None
        # The record's `tokenizer` block: pins the checkpoint name, the
        # tokenizer definition bytes, and the chat template in effect.
        self.tokenizer_info: dict[str, str] = {
            "checkpoint": str(getattr(args, "hf_checkpoint", "") or ""),
            "tokenizer_sha256": _tokenizer_sha256(tokenizer),
            "chat_template_sha256": _chat_template_sha256(tokenizer, tito_tokenizer),
        }

    def create_session(self) -> str:
        self.sweep()
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = LinearTrajectory()
        return session_id

    def get_session(self, session_id: str) -> LinearTrajectory:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.last_used = time.monotonic()
        self.sessions.move_to_end(session_id)
        return session

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        if self.record_sink is not None:
            self.record_sink.finalize(session_id, reason="deleted")

    def sweep(self) -> list[str]:
        """Evict expired-idle sessions (TTL) and least-recently-used overflow
        (max-sessions). Returns the evicted ids. Skips any session that is
        in-flight, locked, or closing — capacity may transiently overflow
        rather than ever killing a live rollout."""
        evicted: list[str] = []
        ttl = self.session_ttl_seconds
        if ttl is not None:
            now = time.monotonic()
            for session_id, session in list(self.sessions.items()):
                if now - session.last_used < ttl:
                    break  # LRU order: everything later was used more recently
                if self._evict(session_id, session, reason="ttl_evicted"):
                    evicted.append(session_id)
        if self.max_sessions is not None:
            overflow = len(self.sessions) - self.max_sessions
            if overflow > 0:
                for session_id, session in list(self.sessions.items()):
                    if overflow <= 0:
                        break
                    if self._evict(session_id, session, reason="capacity_evicted"):
                        evicted.append(session_id)
                        overflow -= 1
        return evicted

    def _evict(self, session_id: str, session: LinearTrajectory, *, reason: str) -> bool:
        if session.inflight > 0 or session.lock.locked() or session.closing:
            return False
        # Flush + finalize the durable record BEFORE the session becomes
        # unreachable — eviction must never lose committed capture.
        if self.record_sink is not None:
            self.record_sink.finalize(session_id, reason=reason)
        self.sessions.pop(session_id, None)
        logger.info("evicted session %s (%s)", session_id, reason)
        if self.on_evict is not None:
            try:
                self.on_evict(session_id)
            except Exception:
                logger.exception("on_evict callback failed for session %s", session_id)
        return True

    def close(self) -> None:
        """Finalize every open record file (process shutdown)."""
        if self.record_sink is not None:
            self.record_sink.close_all(reason="shutdown")

    def compute_session_mismatch(self, session: LinearTrajectory) -> list[dict] | None:
        """Compare accumulated token IDs against a from-scratch render. Read-only."""
        if not session.token_ids:
            return None
        try:
            tools = session.records[-1].request.get("tools") if session.records else None
            expected_ids = self.tito_tokenizer.render_messages(
                session.messages, tools=tools, add_generation_prompt=False, tokenize=True
            )
            mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(f"failed to compute tito_session_mismatch: {e}") from e


def _chat_template_sha256(tokenizer: Any, tito_tokenizer: TITOTokenizer) -> str:
    """SHA-256 of the chat template actually in effect for this gateway —
    the fixed-template override when set (e.g. qwen3_fixed.jinja), otherwise
    the tokenizer's own template. Pins the exact render rules a record's
    token ids were produced under."""
    template = tito_tokenizer.chat_template_kwargs.get("chat_template") or getattr(
        tokenizer, "chat_template", None
    )
    return hashlib.sha256(str(template or "").encode()).hexdigest()


def _tokenizer_sha256(tokenizer: Any) -> str:
    """SHA-256 over the tokenizer DEFINITION bytes — the identity a record's
    token ids depend on.

    Method (documented as part of the record contract): for fast tokenizers,
    hash ``backend_tokenizer.to_str()`` (the complete serialized
    ``tokenizer.json`` definition: vocab, merges, normalizer, added tokens —
    deterministic for a given tokenizer); for slow tokenizers, hash the
    sorted-JSON vocabulary. Two gateways report the same value iff they
    tokenize identically."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            return hashlib.sha256(backend.to_str().encode()).hexdigest()
        except Exception:  # pragma: no cover - serialization quirks per version
            logger.exception("tokenizer_sha256: backend serialization failed; falling back to vocab")
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else {}
    blob = json.dumps(vocab, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
