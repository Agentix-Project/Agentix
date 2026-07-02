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
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .compare import TokenSeqComparator
from .errors import MessageValidationError, SessionNotFoundError, TokenizationError
from .messages import assert_messages_append_only_with_allowed_role, message_matches
from .pretokenize import TITOTokenizer

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


@dataclass
class LinearTrajectory:
    """Message history + accumulated token-ID checkpoints for one session."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
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
        if not self.messages:
            return tito_tokenizer.render_messages(
                request_messages, tools=tools, add_generation_prompt=True, tokenize=True
            )

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
            return tito_tokenizer.render_messages(
                request_messages, tools=tools, add_generation_prompt=True, tokenize=True
            )
        return tito_tokenizer.merge_tokens(
            old_messages=self.messages,
            new_messages=request_messages,
            pretokenized_token_ids=self.token_ids,
            tools=tools,
        )

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
    """Session ID -> trajectory map + shared tokenizer/comparator. Pure CRUD plus the
    read-only mismatch computation; never mutates trajectory state itself."""

    def __init__(self, args: Any, tokenizer: Any, *, tito_tokenizer: TITOTokenizer) -> None:
        self.sessions: dict[str, LinearTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator: TokenSeqComparator = tito_tokenizer.create_comparator()

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = LinearTrajectory()
        return session_id

    def get_session(self, session_id: str) -> LinearTrajectory:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        return session

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

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
