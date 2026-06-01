"""Pure Anthropic ↔ OpenAI shape converters.

Used only by `clients.anthropic_from_openai`. The functions here are
JSON-in, JSON-out — no I/O, no SDK calls, no spans. Anyone writing a
custom Anthropic-on-OpenAI client can import these directly.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any


def anthropic_messages_to_openai(
    body: dict[str, Any],
    *,
    upstream_model: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an Anthropic Messages request body to OpenAI Chat Completions.

    `upstream_model` overrides the Anthropic-side model so the
    OpenAI-compatible upstream receives whatever model name *it*
    understands. `extra_body` is merged into the result for any custom
    upstream fields (reasoning effort, response format, etc.).
    """
    messages: list[dict[str, Any]] = []

    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "\n".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if text:
            messages.append({"role": "system", "content": text})

    messages.extend(_messages_anthropic_to_openai(body.get("messages") or []))

    out: dict[str, Any] = {
        "model": upstream_model or body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
    }
    for key in ("temperature", "top_p", "stop"):
        if key in body:
            out[key] = body[key]

    tools = _tools_anthropic_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools

    if extra_body:
        out.update(extra_body)

    return out


def openai_to_anthropic_messages(
    openai_body: dict[str, Any],
    *,
    response_model: str,
) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages."""
    choice = (openai_body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = openai_body.get("usage") or {}
    content: list[dict[str, Any]] = []

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name", ""),
                "input": args,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    finish_reason = choice.get("finish_reason")
    stop_reason = "end_turn"
    if finish_reason == "tool_calls" or any(block["type"] == "tool_use" for block in content):
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": response_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
    }


def anthropic_sse(body: dict[str, Any]) -> bytes:
    """Render an Anthropic Messages response as a single-shot SSE blob.

    Some Anthropic SDKs always open a streaming connection; replaying a
    completed response as a sequence of `message_start` /
    `content_block_*` / `message_stop` events gives them a valid wire
    stream without actually streaming from the upstream.
    """
    content = body.get("content") or []
    usage = body.get("usage") or {}
    parts: list[bytes] = [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": body["id"],
                    "type": "message",
                    "role": "assistant",
                    "model": body.get("model", ""),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": 0,
                    },
                },
            },
        )
    ]

    for index, block in enumerate(content):
        block_type = block.get("type")
        if block_type == "text":
            parts.append(
                _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            text = block.get("text", "")
            if text:
                parts.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                )
            parts.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        elif block_type == "tool_use":
            parts.append(
                _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.get("id"),
                            "name": block.get("name", ""),
                            "input": {},
                        },
                    },
                )
            )
            partial_json = json.dumps(block.get("input") or {}, separators=(",", ":"))
            parts.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": partial_json},
                    },
                )
            )
            parts.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))

    parts.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": body.get("stop_reason", "end_turn"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    parts.append(_sse("message_stop", {"type": "message_stop"}))
    return b"".join(parts)


@dataclasses.dataclass(slots=True)
class AnthropicCountTokens:
    input_tokens: int


def count_anthropic_tokens(body: dict[str, Any]) -> AnthropicCountTokens:
    """Approximate input-token count for `/v1/messages/count_tokens`.

    A 1-char-per-4-bytes estimate good enough for capture/replay; keeps
    the adapter standalone (no tokenizer dependency)."""
    chars = 0
    system = body.get("system")
    if isinstance(system, str):
        chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                chars += len(str(block.get("text", "")))
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    chars += len(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    text = block.get("content")
                    if isinstance(text, str):
                        chars += len(text)
                    elif isinstance(text, list):
                        for sub in text:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                chars += len(str(sub.get("text", "")))
    return AnthropicCountTokens(input_tokens=max(1, chars // 4))


# ── internals ─────────────────────────────────────────────────────────────


def _messages_anthropic_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif block_type == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _tool_result_text(block.get("content")),
                    }
                )

        text = "\n".join(part for part in text_parts if part)
        if role == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            if message["content"] is not None or tool_calls:
                out.append(message)
        else:
            out.extend(tool_results)
            if text:
                out.append({"role": role, "content": text})
    return out


def _tools_anthropic_to_openai(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or "name" not in tool:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {},
                },
            }
        )
    return out


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _sse(event: str, data: dict[str, Any]) -> bytes:
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()


__all__ = [
    "AnthropicCountTokens",
    "anthropic_messages_to_openai",
    "anthropic_sse",
    "count_anthropic_tokens",
    "openai_to_anthropic_messages",
]
