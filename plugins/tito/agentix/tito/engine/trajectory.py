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


@dataclass
class LinearTrajectory:
    """Message history + accumulated token-ID checkpoints for one session."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
    records: list[SessionRecord] = field(default_factory=list)
    trajectory_token_ids: list[list[int]] = field(default_factory=list)
    num_assistant: int = 0

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
        if not self.token_ids:
            return tito_tokenizer.render_messages(
                request_messages, tools=tools, add_generation_prompt=True, tokenize=True
            )

        self._try_detect_and_rollback_to_assistant_checkpoint(request_messages)
        try:
            assert_messages_append_only_with_allowed_role(
                self.messages, request_messages, tito_tokenizer.allowed_append_roles
            )
        except ValueError as e:
            raise MessageValidationError(f"{e}; to allow more roles use --tito-allowed-append-roles") from e

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
                first_mismatch = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=True))
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

    def _try_detect_and_rollback_to_assistant_checkpoint(self, request_messages: list[dict[str, Any]]) -> None:
        """If *request_messages* diverges from the stored history, truncate state back to
        the last assistant checkpoint within the matching prefix (single-step only)."""
        stored = self.messages
        if not stored or not self.trajectory_token_ids:
            return

        match_len = 0
        for i in range(min(len(request_messages), len(stored))):
            if message_matches(stored[i], request_messages[i]):
                match_len = i + 1
            else:
                break

        if match_len >= len(stored):
            return

        rollback_msg_end = None
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

        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> checkpoint %d (messages[:%d]), "
            "discarding %d assistant(s)",
            len(stored), self.num_assistant, checkpoint_index, rollback_msg_end, discard_count,
        )
        self.messages = stored[:rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: checkpoint_index + 1]
        self.records = self.records[: checkpoint_index + 1]
        self.num_assistant = checkpoint_index + 1


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
