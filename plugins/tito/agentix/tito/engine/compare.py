"""Token-sequence comparator: segment by special tokens, classify mismatches.

Used to check that an incrementally-accumulated trajectory tokenizes identically
to a from-scratch render. The comparison is structural: the special-token skeleton
and non-assistant content must match exactly; assistant content may differ (the
model's own tokens) and is reported as a soft mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MismatchType(StrEnum):
    # Segment count or special/content pattern differs — structural break.
    SPECIAL_TOKEN_COUNT = "special_token_count"
    # Aligned special-token segment holds a different special token.
    SPECIAL_TOKEN_TYPE = "special_token_type"
    # Non-assistant content (system/user/tool) differs — the prompt drifted.
    NON_ASSISTANT_TEXT = "non_assistant_text"
    # Assistant content differs — expected and non-severe (model's own tokens).
    ASSISTANT_TEXT = "assistant_text"


@dataclass
class Segment:
    token_ids: list[int]
    is_special: bool = False


@dataclass
class Mismatch:
    type: MismatchType
    segment_index: int
    expected_text: str = ""
    actual_text: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "segment_index": self.segment_index,
            "expected_text": self.expected_text,
            "actual_text": self.actual_text,
            "detail": self.detail,
        }


class TokenSeqComparator:
    """Segment two token-ID sequences at special-token boundaries and compare.

    `assistant_start_str` (e.g. ``"<|im_start|>assistant"``) classifies a content
    segment as assistant vs non-assistant. `special_token_ids`, if given, overrides
    the set collected from the tokenizer. `trim_trailing_ids` are stripped from both
    tails before comparison (a stop token the model emits but the template doesn't).
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        assistant_start_str: str | None,
        special_token_ids: set[int] | None = None,
        trim_trailing_ids: frozenset[int] | set[int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._assistant_start_str = assistant_start_str
        self._special_ids = (
            set(special_token_ids) if special_token_ids is not None else self.collect_special_ids(tokenizer)
        )
        self._trim_trailing_ids = set(trim_trailing_ids) if trim_trailing_ids else None

    @staticmethod
    def collect_special_ids(tokenizer: Any) -> set[int]:
        """Token IDs flagged ``special=True`` by the tokenizer. Content tokens a role
        produces (e.g. ``<think>``) are NOT special, so they aren't collected here."""
        ids: set[int] = set(getattr(tokenizer, "all_special_ids", []) or [])
        decoder = getattr(tokenizer, "added_tokens_decoder", None)
        if decoder:
            ids |= {k for k, v in decoder.items() if getattr(v, "special", False)}
        return ids

    def segment_by_special_tokens(self, token_ids: list[int]) -> list[Segment]:
        """Each special token is its own single-ID segment; consecutive non-special
        tokens group into one content segment."""
        segments: list[Segment] = []
        current: list[int] = []
        for tid in token_ids:
            if tid in self._special_ids:
                if current:
                    segments.append(Segment(token_ids=current))
                    current = []
                segments.append(Segment(token_ids=[tid], is_special=True))
            else:
                current.append(tid)
        if current:
            segments.append(Segment(token_ids=current))
        return segments

    def compare_sequences(
        self,
        expected_ids: list[int],
        actual_ids: list[int],
        trim_trailing_ids: frozenset[int] | set[int] | None = None,
    ) -> list[Mismatch]:
        trim = self._trim_trailing_ids or set()
        if trim_trailing_ids:
            trim = trim | trim_trailing_ids
        if trim:
            expected_ids = _trim_trailing(expected_ids, trim)
            actual_ids = _trim_trailing(actual_ids, trim)

        exp_segs = self.segment_by_special_tokens(expected_ids)
        act_segs = self.segment_by_special_tokens(actual_ids)

        structural = self._check_segment_structure(exp_segs, act_segs)
        if structural is not None:
            return [structural]

        mismatches: list[Mismatch] = []
        for idx, (exp, act) in enumerate(zip(exp_segs, act_segs, strict=True)):
            is_assistant = self._is_assistant_content(exp_segs, idx) and self._is_assistant_content(act_segs, idx)
            m = self._compare_single_segment(idx, exp, act, is_assistant_content=is_assistant)
            if m is not None:
                mismatches.append(m)
        return mismatches

    def _check_segment_structure(self, exp_segs: list[Segment], act_segs: list[Segment]) -> Mismatch | None:
        if len(exp_segs) != len(act_segs):
            detail = f"segment count differs: expected {len(exp_segs)}, got {len(act_segs)}"
        elif [s.is_special for s in exp_segs] != [s.is_special for s in act_segs]:
            detail = "segment structure (special/content pattern) differs"
        else:
            return None
        return Mismatch(
            type=MismatchType.SPECIAL_TOKEN_COUNT,
            segment_index=-1,
            expected_text=self._describe_structure(exp_segs),
            actual_text=self._describe_structure(act_segs),
            detail=detail,
        )

    def _compare_single_segment(
        self, idx: int, exp: Segment, act: Segment, *, is_assistant_content: bool
    ) -> Mismatch | None:
        if exp.is_special:
            if exp.token_ids != act.token_ids:
                return Mismatch(
                    type=MismatchType.SPECIAL_TOKEN_TYPE,
                    segment_index=idx,
                    expected_text=self._decode(exp.token_ids),
                    actual_text=self._decode(act.token_ids),
                )
            return None
        exp_text = self._decode(exp.token_ids)
        act_text = self._decode(act.token_ids)
        if exp_text == act_text:
            return None
        return Mismatch(
            type=MismatchType.ASSISTANT_TEXT if is_assistant_content else MismatchType.NON_ASSISTANT_TEXT,
            segment_index=idx,
            expected_text=exp_text,
            actual_text=act_text,
        )

    def _is_assistant_content(self, segments: list[Segment], idx: int) -> bool:
        if self._assistant_start_str is None:
            return False
        if segments[idx].is_special or idx == 0:
            return False
        prev = segments[idx - 1]
        if not prev.is_special:
            return False
        special_text = self._decode(prev.token_ids)
        content_prefix = self._decode(segments[idx].token_ids[:20])
        return (special_text + content_prefix).startswith(self._assistant_start_str)

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _describe_structure(self, segments: list[Segment]) -> str:
        return " ".join(
            f"[{self._decode(s.token_ids)}]" if s.is_special else f"({len(s.token_ids)} tokens)" for s in segments
        )


def _trim_trailing(ids: list[int], to_remove: set[int]) -> list[int]:
    end = len(ids)
    while end > 0 and ids[end - 1] in to_remove:
        end -= 1
    return ids[:end]
