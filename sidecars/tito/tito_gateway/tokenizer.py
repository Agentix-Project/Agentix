"""Public tokenizer entrypoints for the vendored Miles TITO implementation."""

from __future__ import annotations

from enum import Enum
from typing import Any


class TITOTokenizerType(str, Enum):
    DEFAULT = "default"
    QWEN3 = "qwen3"
    QWEN35 = "qwen35"
    QWENNEXT = "qwennext"
    GLM47 = "glm47"
    NEMOTRON3 = "nemotron3"
    KIMI25 = "kimi25"
    KIMI26 = "kimi26"
    MINIMAX_M25 = "minimax_m25"
    MINIMAX_M27 = "minimax_m27"
    DEEPSEEKV32 = "deepseekv32"
    DEEPSEEKV4 = "deepseekv4"


def get_tito_tokenizer(*args: Any, **kwargs: Any) -> Any:
    """Return a vendored Miles TITO tokenizer instance."""
    from tito_gateway.vendor.miles_compat.utils.chat_template_utils import (
        get_tito_tokenizer as _get_tito_tokenizer,
    )

    return _get_tito_tokenizer(*args, **kwargs)
