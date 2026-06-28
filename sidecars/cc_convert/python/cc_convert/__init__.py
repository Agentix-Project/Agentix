"""cc_convert — Anthropic ↔ OpenAI Chat Completions protocol converter.

The heavy lifting is implemented in Rust and exposed through the native
extension `cc_convert._native`. This module provides ergonomic Python wrappers
that work with dicts (instead of JSON strings).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from . import _native  # type: ignore

__version__: str = _native.__version__
__all__ = [
    "translate_request",
    "translate_response",
    "StreamTranslator",
    "__version__",
]


def translate_request(
    anthropic_request: Dict[str, Any],
    target_model: Optional[str] = None,
    mode: str = "pragmatic",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Translate an Anthropic Messages request dict into an OpenAI Chat
    Completions request dict.

    Args:
        anthropic_request: the Anthropic-shape request body.
        target_model: override the ``model`` field in the translated request.
        mode: ``"pragmatic"`` (default) collapses single-text content to a
            string, drops top_k, injects stream_options.include_usage, etc.
            ``"litellm_compat"`` matches LiteLLM's AnthropicAdapter byte-for-byte
            (useful as a drop-in replacement in an existing LiteLLM pipeline).

    Returns ``(openai_request, tool_name_map)``. Keep the ``tool_name_map`` and
    pass it to :func:`translate_response` / :class:`StreamTranslator` so we can
    restore tool names that had to be truncated to fit OpenAI's 64-char limit.
    """

    openai_str, map_str = _native.translate_request(
        json.dumps(anthropic_request), target_model, mode
    )
    return json.loads(openai_str), json.loads(map_str)


def translate_response(
    openai_response: Dict[str, Any],
    original_model: str,
    tool_name_map: Dict[str, str],
) -> Dict[str, Any]:
    """Translate an OpenAI Chat Completions response dict into an Anthropic
    Messages response dict.
    """

    anthropic_str = _native.translate_response(
        json.dumps(openai_response), original_model, json.dumps(tool_name_map)
    )
    return json.loads(anthropic_str)


class StreamTranslator:
    """Stateful translator: feed it OpenAI SSE chunk dicts and read back
    Anthropic SSE event dicts. Call :meth:`finish` when upstream is exhausted.
    """

    def __init__(self, original_model: str, tool_name_map: Dict[str, str]) -> None:
        self._inner = _native.PyStreamTranslator(
            original_model, json.dumps(tool_name_map)
        )

    def push(self, openai_chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        events_json = self._inner.push(json.dumps(openai_chunk))
        return [json.loads(e) for e in events_json]

    def finish(self) -> List[Dict[str, Any]]:
        events_json = self._inner.finish()
        return [json.loads(e) for e in events_json]
