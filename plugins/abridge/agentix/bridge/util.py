"""Shared bridge utilities."""

from __future__ import annotations

import base64
import contextlib
import json
import os
from typing import Any


def csv_env(name: str, default: str) -> list[str]:
    return [part.strip() for part in os.getenv(name, default).split(",") if part.strip()]


def host_matches(host: str, patterns: list[str]) -> bool:
    host_without_port = host.rsplit(":", 1)[0] if ":" in host and not host.startswith("[") else host
    return any(
        host == pattern
        or host.endswith(f".{pattern}")
        or host_without_port == pattern
        or host_without_port.endswith(f".{pattern}")
        for pattern in patterns
    )


def join_url_path(base_path: str, suffix: str) -> str:
    left = base_path.rstrip("/")
    right = suffix.lstrip("/")
    if not left:
        return f"/{right}"
    return f"{left}/{right}"


def trace(event: str, **fields: Any) -> None:
    if os.getenv("ABRIDGE_TRACE", "1") in {"0", "false", "False", "no"}:
        return
    print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)


def content_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        raw = content.encode()
    elif isinstance(content, bytes):
        raw = content
    elif isinstance(content, bytearray | memoryview):
        raw = bytes(content)
    elif content is None:
        raw = b""
    else:
        raw = str(content).encode()

    payload: dict[str, Any] = {
        "bytes": len(raw),
        "body_base64": base64.b64encode(raw).decode(),
    }
    with contextlib.suppress(UnicodeDecodeError):
        payload["text"] = raw.decode()
    return payload


def sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def message_text(message: Any) -> str:
    try:
        return message.get_text(strict=False)
    except TypeError:
        return message.get_text()


def tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)
