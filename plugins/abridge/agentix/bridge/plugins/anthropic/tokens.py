"""Anthropic count_tokens stub."""

from __future__ import annotations

import json
from typing import Any


def count_tokens(body: dict[str, Any]) -> dict[str, int]:
    messages = body.get("messages") or []
    return {"input_tokens": len(json.dumps(messages, separators=(",", ":"))) // 4}
