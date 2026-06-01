"""OTel GenAI span attribute helpers shared by the bundled clients.

Bundled clients call `populate_openai_span` / `populate_anthropic_span`
from their `@on(...)` methods so OTel backends (LangSmith, Langfuse,
Datadog, …) render the call as an LLM operation out of the box. Custom
handlers can call them too, or set their own attrs directly via
`trace.get_current_span().set_attribute(...)` — abridge's `Proxy`
itself never inspects request/response shape.

All attrs use `set_attribute` (idempotent / last-write-wins), so a
handler that wraps another (e.g. `AnthropicFromOpenAIClient` over the
OpenAI SDK) can populate AFTER the inner call and the agent-facing
shape wins on the span.
"""

from __future__ import annotations

import json
from typing import Any

from agentix.utils import trace


def populate_openai_span(*, request: dict[str, Any], response: dict[str, Any]) -> None:
    """Stamp OTel GenAI attrs on the current `/trace` span for an OpenAI
    Chat Completions call. No-op when no span is open."""
    sp = trace.get_current_span()
    if sp is None:
        return
    _populate_request_attrs(sp, system="openai", request=request)
    _populate_openai_response_attrs(sp, response)


def populate_anthropic_span(*, request: dict[str, Any], response: dict[str, Any]) -> None:
    """Anthropic version of `populate_openai_span`."""
    sp = trace.get_current_span()
    if sp is None:
        return
    _populate_request_attrs(sp, system="anthropic", request=request)
    _populate_anthropic_response_attrs(sp, response)


def _populate_request_attrs(sp: trace.Span, *, system: str, request: dict[str, Any]) -> None:
    sp.set_attributes(
        **{
            "gen_ai.operation.name": "chat",
            "gen_ai.system": system,
            "gen_ai.request.model": str(request.get("model") or ""),
        }
    )
    for key, attr in (
        ("max_tokens", "gen_ai.request.max_tokens"),
        ("temperature", "gen_ai.request.temperature"),
        ("top_p", "gen_ai.request.top_p"),
    ):
        if request.get(key) is not None:
            sp.set_attribute(attr, request[key])

    idx = 0
    system_prompt = request.get("system")
    if system_prompt:
        sp.set_attribute("gen_ai.prompt.0.role", "system")
        sp.set_attribute("gen_ai.prompt.0.content", _content_to_text(system_prompt))
        idx = 1
    for msg in request.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        sp.set_attribute(f"gen_ai.prompt.{idx}.role", str(msg.get("role", "user")))
        sp.set_attribute(f"gen_ai.prompt.{idx}.content", _content_to_text(msg.get("content", "")))
        idx += 1


def _populate_openai_response_attrs(sp: trace.Span, response: dict[str, Any]) -> None:
    u = response.get("usage") or {}
    sp.set_attributes(
        **{
            "gen_ai.usage.input_tokens": int(u.get("prompt_tokens") or 0),
            "gen_ai.usage.output_tokens": int(u.get("completion_tokens") or 0),
        }
    )
    model = response.get("model")
    if isinstance(model, str) and model:
        sp.set_attribute("gen_ai.response.model", model)
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") if isinstance(choice, dict) else None
    sp.set_attribute("gen_ai.completion.0.role", "assistant")
    sp.set_attribute("gen_ai.completion.0.content", str((msg or {}).get("content") or ""))
    names: list[str] = []
    for tc in (msg or {}).get("tool_calls") or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        name = (fn or {}).get("name")
        if isinstance(name, str):
            names.append(name)
    if names:
        sp.set_attribute("gen_ai.tool_calls.names", names)


def _populate_anthropic_response_attrs(sp: trace.Span, response: dict[str, Any]) -> None:
    u = response.get("usage") or {}
    sp.set_attributes(
        **{
            "gen_ai.usage.input_tokens": int(u.get("input_tokens") or 0),
            "gen_ai.usage.output_tokens": int(u.get("output_tokens") or 0),
        }
    )
    model = response.get("model")
    if isinstance(model, str) and model:
        sp.set_attribute("gen_ai.response.model", model)
    sp.set_attribute("gen_ai.completion.0.role", "assistant")
    sp.set_attribute("gen_ai.completion.0.content", _content_to_text(response.get("content")))
    names: list[str] = []
    for block in response.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name")
            if isinstance(name, str):
                names.append(name)
    if names:
        sp.set_attribute("gen_ai.tool_calls.names", names)


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic/OpenAI message content (str or block list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                bt = block.get("type")
                if bt == "text":
                    parts.append(str(block.get("text", "")))
                elif bt == "tool_use":
                    parts.append(
                        f"[tool_use {block.get('name', '')} {json.dumps(block.get('input') or {})}]"
                    )
                elif bt == "tool_result":
                    parts.append(f"[tool_result] {_content_to_text(block.get('content'))}")
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


__all__ = ["populate_anthropic_span", "populate_openai_span"]
