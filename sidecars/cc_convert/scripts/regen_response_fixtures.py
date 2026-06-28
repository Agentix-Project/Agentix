"""Regenerate Anthropic-shape golden response files from OpenAI inputs
using LiteLLM as the oracle.

For each fixture under tests/fixtures/responses/openai_<name>.json, this
loads the corresponding meta_<name>.json (which carries any tool_map) and
runs LiteLLM's translate_openai_response_to_anthropic, writing
anthropic_<name>.json next to the input.

Run:

    pip install 'litellm>=1.0'
    python scripts/regen_response_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "responses"


def _import_litellm():
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.types.utils import (
            ChatCompletionMessageToolCall,
            Choices,
            Function,
            Message,
            ModelResponse,
            PromptTokensDetailsWrapper,
            Usage,
        )
    except ImportError as e:
        print(f"litellm missing: {e}", file=sys.stderr)
        sys.exit(1)
    return (
        LiteLLMAnthropicMessagesAdapter(),
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    )


def _build_model_response(
    raw: Dict[str, Any],
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
):
    """Translate a JSON OpenAI ChatCompletion dict into the LiteLLM
    ModelResponse object the adapter expects."""

    def build_message(m: Dict[str, Any]) -> Any:
        kw: Dict[str, Any] = {"role": m.get("role", "assistant")}
        if m.get("content") is not None:
            kw["content"] = m["content"]
        if m.get("reasoning_content") is not None:
            kw["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            kw["tool_calls"] = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=Function(
                        name=tc["function"]["name"],
                        arguments=tc["function"].get("arguments", ""),
                    ),
                )
                for tc in m["tool_calls"]
            ]
        return Message(**kw)

    choices = [
        Choices(
            index=c.get("index", 0),
            message=build_message(c["message"]),
            finish_reason=c.get("finish_reason"),
        )
        for c in raw["choices"]
    ]
    usage_kw: Dict[str, Any] = {}
    if (u := raw.get("usage")):
        usage_kw["prompt_tokens"] = u.get("prompt_tokens", 0)
        usage_kw["completion_tokens"] = u.get("completion_tokens", 0)
        if (ptd := u.get("prompt_tokens_details")):
            usage_kw["prompt_tokens_details"] = PromptTokensDetailsWrapper(
                cached_tokens=ptd.get("cached_tokens", 0)
            )
    return ModelResponse(
        id=raw["id"],
        model=raw.get("model", "gpt-4o-mini"),
        choices=choices,
        usage=Usage(**usage_kw),
    )


def main() -> None:
    (
        adapter,
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
        ModelResponse,
        PromptTokensDetailsWrapper,
        Usage,
    ) = _import_litellm()

    n = 0
    for input_path in sorted(ROOT.glob("openai_*.json")):
        name = input_path.stem.removeprefix("openai_")
        meta_path = ROOT / f"meta_{name}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        raw = json.loads(input_path.read_text())
        try:
            model_resp = _build_model_response(
                raw,
                ChatCompletionMessageToolCall,
                Choices,
                Function,
                Message,
                ModelResponse,
                PromptTokensDetailsWrapper,
                Usage,
            )
            tool_map = meta.get("tool_map") or {}
            golden = adapter.translate_openai_response_to_anthropic(
                model_resp, tool_name_mapping=tool_map
            )
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        # LiteLLM returns a TypedDict; dump with default=str for safety.
        out_path = ROOT / f"anthropic_{name}.json"
        out_path.write_text(
            json.dumps(golden, indent=2, sort_keys=True, default=str) + "\n"
        )
        n += 1
        print(f"  ✓ {name}")
    print(f"\nregenerated {n} response goldens via LiteLLM")


if __name__ == "__main__":
    main()
