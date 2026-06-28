"""Regenerate Anthropic-shape stream goldens from OpenAI .sse inputs using
LiteLLM's AnthropicStreamWrapper.

Reads each tests/fixtures/streams/openai_<name>.sse, parses the chunks as
ModelResponse-equivalent objects, feeds them into AnthropicStreamWrapper,
and writes the resulting Anthropic events (one per line) to
anthropic_<name>.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "streams"


def _import_litellm():
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
            AnthropicStreamWrapper,
        )
        from litellm.types.utils import (
            ChatCompletionDeltaToolCall,
            Delta,
            Function,
            ModelResponse,
            ModelResponseStream,
            StreamingChoices,
            Usage,
        )
    except ImportError as e:
        print(f"litellm missing: {e}", file=sys.stderr)
        sys.exit(1)
    return (
        AnthropicStreamWrapper,
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponse,
        ModelResponseStream,
        StreamingChoices,
        Usage,
    )


def parse_sse(text: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    chunks.append(json.loads(payload))
    return chunks


def build_chunk(
    raw: Dict[str, Any],
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
):
    def build_delta(d: Dict[str, Any]) -> Any:
        kw: Dict[str, Any] = {}
        if d.get("role") is not None:
            kw["role"] = d["role"]
        if d.get("content") is not None:
            kw["content"] = d["content"]
        if d.get("reasoning_content") is not None:
            kw["reasoning_content"] = d["reasoning_content"]
        if d.get("tool_calls"):
            kw["tool_calls"] = [
                ChatCompletionDeltaToolCall(
                    id=tc.get("id"),
                    type=tc.get("type", "function"),
                    index=tc.get("index", 0),
                    function=Function(
                        name=tc.get("function", {}).get("name"),
                        arguments=tc.get("function", {}).get("arguments"),
                    ),
                )
                for tc in d["tool_calls"]
            ]
        return Delta(**kw)

    choices = [
        StreamingChoices(
            index=c.get("index", 0),
            delta=build_delta(c.get("delta", {})),
            finish_reason=c.get("finish_reason"),
        )
        for c in raw.get("choices", [])
    ]
    kw: Dict[str, Any] = {"id": raw.get("id"), "choices": choices}
    if raw.get("usage"):
        kw["usage"] = Usage(**raw["usage"])
    return ModelResponseStream(**kw)


def main() -> None:
    (
        AnthropicStreamWrapper,
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponse,
        ModelResponseStream,
        StreamingChoices,
        Usage,
    ) = _import_litellm()

    n = 0
    for input_path in sorted(ROOT.glob("openai_*.sse")):
        name = input_path.stem.removeprefix("openai_")
        raw_chunks = parse_sse(input_path.read_text())
        try:
            chunks = [
                build_chunk(
                    c,
                    ChatCompletionDeltaToolCall,
                    Delta,
                    Function,
                    ModelResponseStream,
                    StreamingChoices,
                    Usage,
                )
                for c in raw_chunks
            ]
            wrapper = AnthropicStreamWrapper(
                completion_stream=iter(chunks),
                model="claude-opus-4-7",
            )
            events = list(wrapper)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e!r}", file=sys.stderr)
            continue
        out_path = ROOT / f"anthropic_{name}.jsonl"
        with out_path.open("w") as f:
            for ev in events:
                f.write(json.dumps(ev, default=str, sort_keys=True) + "\n")
        n += 1
        print(f"  ✓ {name}  ({len(events)} events)")
    print(f"\nregenerated {n} stream goldens via LiteLLM")


if __name__ == "__main__":
    main()
