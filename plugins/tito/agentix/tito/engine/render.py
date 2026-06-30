"""Chat-template rendering backend.

`apply_chat_template` renders messages through an HF tokenizer's chat template
(optionally an explicit `chat_template=` string for the fixed template), the same
code path SGLang uses. Tool definitions are canonicalized to the OpenAI
`{type:"function", function:{...}}` shape. No sglang dependency — the one pydantic
`Tool` type the canonicalization needs is defined locally.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Literal, Optional

from jinja2 import TemplateError
from pydantic import BaseModel, TypeAdapter


class _Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: _Function


def normalize_tool_arguments(messages: list[dict], format: Literal["dict", "json"]) -> list[dict]:
    """Deep-copy *messages*, set assistant `content: None` -> "", and coerce tool_call
    `arguments` to the form the renderer needs: "dict" (JSON string -> dict, for
    HF-Jinja templates) or "json" (dict -> JSON string). Never mutates the input."""
    normalized = copy.deepcopy(messages)
    for msg in normalized:
        if msg.get("role") == "assistant":
            if msg.get("content") is None:
                msg["content"] = ""
            if isinstance(msg.get("tool_calls"), list):
                for item in msg["tool_calls"]:
                    func = item.get("function")
                    if not func:
                        continue
                    args = func.get("arguments")
                    if format == "dict" and isinstance(args, str):
                        func["arguments"] = json.loads(args)
                    elif format == "json" and isinstance(args, dict):
                        func["arguments"] = json.dumps(args, ensure_ascii=False)
    return normalized


def extract_tool_dicts(tools: list[dict] | None) -> list[dict] | None:
    """Canonicalize tools to full `{type:"function", function:{...}}` dumps."""
    if not tools:
        return None
    wrapped = [t if isinstance(t, dict) and "function" in t else {"type": "function", "function": t} for t in tools]
    validated = TypeAdapter(list[Tool]).validate_python(wrapped)
    return [tool.model_dump() for tool in validated]


def apply_chat_template(
    messages: list[dict],
    *,
    tokenizer: Any,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs: Any,
) -> str | list[int]:
    """Render via the HF tokenizer in SGLang style (`return_dict=False`, so the result
    is `str` when tokenize=False or `list[int]` when tokenize=True). `chat_template=`
    and other extras pass through `**kwargs`. Falls back to the bare function schema if
    the template can't take the wrapped tool dicts."""
    messages = normalize_tool_arguments(messages, "dict")
    tool_defs = extract_tool_dicts(tools)
    render_kwargs = dict(add_generation_prompt=add_generation_prompt, **kwargs)
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, tools=tool_defs, return_dict=False, **render_kwargs
        )
    except TemplateError as e:
        if tool_defs is not None:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    tools=[t["function"] if "function" in t else t for t in tool_defs],
                    return_dict=False,
                    **render_kwargs,
                )
            except TemplateError as te:
                raise ValueError(f"Chat template rendering failed (tool format fallback): {te}") from te
        raise ValueError(f"Chat template rendering failed: {e}") from e
