"""Anthropic Messages ↔ OpenAI Chat Completions conversion."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from agentix.bridge.actions import json_envelope, respond_action, response_envelope
from agentix.bridge.plugins.anthropic.tokens import count_tokens
from agentix.bridge.types import ProxyAction, ProxyEvent, ResponseEnvelope
from agentix.bridge.util import sse, tool_result_text


def anthropic_to_openai_body(
    body: dict[str, Any],
    *,
    upstream_model: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
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

    messages.extend(anthropic_messages_to_openai(body.get("messages") or []))

    out: dict[str, Any] = {
        "model": upstream_model or os.getenv("OPENAI_MODEL") or os.getenv("ABRIDGE_OPENAI_MODEL") or body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_tokens", 4096),
    }
    for key in ("temperature", "top_p", "stop"):
        if key in body:
            out[key] = body[key]

    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools

    env_extra_body = os.getenv("ABRIDGE_OPENAI_EXTRA_BODY")
    if env_extra_body:
        extra = json.loads(env_extra_body)
        if not isinstance(extra, dict):
            raise ValueError("ABRIDGE_OPENAI_EXTRA_BODY must decode to a JSON object")
        out.update(extra)
    if extra_body:
        out.update(extra_body)

    return out


def anthropic_messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
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
                out.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input") or {}),
                                },
                            }
                        ],
                    }
                )
            elif block_type == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": tool_result_text(block.get("content")),
                    }
                )
        if text_parts:
            out.append({"role": role, "content": "\n".join(text_parts)})
    return out


def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
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


def openai_to_anthropic_body(openai_body: dict[str, Any], *, response_model: str) -> dict[str, Any]:
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
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def anthropic_sse(body: dict[str, Any]) -> bytes:
    content = body.get("content") or []
    usage = body.get("usage") or {}
    parts = [
        sse(
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
                    "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
                },
            },
        )
    ]

    for index, block in enumerate(content):
        block_type = block.get("type")
        if block_type == "text":
            parts.append(
                sse(
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
                    sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                )
            parts.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))
        elif block_type == "tool_use":
            parts.append(
                sse(
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
                sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": partial_json},
                    },
                )
            )
            parts.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))

    parts.append(
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": body.get("stop_reason", "end_turn"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    parts.append(sse("message_stop", {"type": "message_stop"}))
    return b"".join(parts)


def anthropic_path(body: dict[str, Any], path: str) -> ResponseEnvelope | None:
    if path == "/v1/messages/count_tokens":
        return json_envelope(count_tokens(body))
    if path != "/v1/messages":
        return response_envelope(404, f"unsupported captured path: {path}".encode())
    return None


def anthropic_client_envelope(
    body: dict[str, Any],
    upstream_body: dict[str, Any],
) -> ResponseEnvelope:
    client_stream = bool(body.get("stream"))
    response_model = str(body.get("model") or "")
    anthropic_body = openai_to_anthropic_body(upstream_body, response_model=response_model)
    if client_stream:
        return response_envelope(200, anthropic_sse(anthropic_body), content_type="text/event-stream")
    return json_envelope(anthropic_body)


def forward_anthropic_http_event(
    event: ProxyEvent,
    *,
    upstream_model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    post_openai: Any,
) -> ResponseEnvelope:
    request = event.get("request") or {}
    if not isinstance(request, dict):
        return response_envelope(400, b"bad forwarded request")

    path = str(request.get("path") or "").split("?", 1)[0]
    body = request.get("json") or {}
    if not isinstance(body, dict):
        body = {}

    local = anthropic_path(body, path)
    if local is not None:
        return local

    openai_body = anthropic_to_openai_body(body, upstream_model=upstream_model, extra_body=extra_body)
    openai_body["stream"] = False
    upstream = post_openai(openai_body)
    return anthropic_client_envelope(body, upstream)


async def forward_anthropic_http_event_async(
    event: ProxyEvent,
    *,
    upstream_model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    post_openai: Any,
) -> ResponseEnvelope:
    request = event.get("request") or {}
    if not isinstance(request, dict):
        return response_envelope(400, b"bad forwarded request")

    path = str(request.get("path") or "").split("?", 1)[0]
    body = request.get("json") or {}
    if not isinstance(body, dict):
        body = {}

    local = anthropic_path(body, path)
    if local is not None:
        return local

    openai_body = anthropic_to_openai_body(body, upstream_model=upstream_model, extra_body=extra_body)
    openai_body["stream"] = False
    upstream = await post_openai(openai_body)
    return anthropic_client_envelope(body, upstream)


def anthropic_http_action(
    event: ProxyEvent,
    *,
    upstream_model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    post_openai: Any,
) -> ProxyAction:
    return respond_action(
        forward_anthropic_http_event(
            event,
            upstream_model=upstream_model,
            extra_body=extra_body,
            post_openai=post_openai,
        )
    )
