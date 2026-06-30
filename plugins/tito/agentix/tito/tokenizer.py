"""Public tokenizer entrypoints — thin re-export of the native TITO engine."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from .engine.pretokenize import get_tito_tokenizer as _engine_get_tito_tokenizer


class TITOTokenizerType(StrEnum):
    """Tokenizer families the native engine supports. Other models are a small
    subclass + a fixed chat template — see agentix.tito.engine.pretokenize."""

    DEFAULT = "default"
    QWEN3 = "qwen3"


def get_tito_tokenizer(
    tokenizer: Any,
    tokenizer_type: TITOTokenizerType | str = TITOTokenizerType.DEFAULT,
    *,
    allowed_append_roles: tuple[str, ...] | list[str] | None = None,
    **_ignored: Any,
) -> Any:
    """Build a TITO tokenizer for *tokenizer* (`"qwen3"` or `"default"`)."""
    t = tokenizer_type.value if isinstance(tokenizer_type, TITOTokenizerType) else str(tokenizer_type)
    roles = tuple(allowed_append_roles) if allowed_append_roles else ("tool",)
    return _engine_get_tito_tokenizer(tokenizer, t, allowed_append_roles=roles)
