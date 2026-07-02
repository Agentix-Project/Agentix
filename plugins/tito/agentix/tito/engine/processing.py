"""Tokenizer loading. Minimal: load an HF tokenizer (tokenizer-only is fine — no
torch needed) and optionally override its chat template from a file."""

from __future__ import annotations

from typing import Any


def load_tokenizer(name_or_path: str, chat_template_path: str | None = None, *, trust_remote_code: bool = False) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
    if chat_template_path:
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
    return tokenizer
