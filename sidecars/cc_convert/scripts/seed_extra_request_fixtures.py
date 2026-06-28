"""Add request fixtures 32-43 covering message shapes that the first batch missed.

Cases:
  32_agent_tool_loop                  multi-turn: user → assistant tool_use → user tool_result → assistant text
  33_user_content_cache_control       cache_control on a user text block
  34_assistant_content_cache_control  cache_control on an assistant text block
  35_assistant_thinking_history       prior assistant turn with `thinking` block
  36_user_mixed_content               text + image + tool_result in same user message
  37_empty_string_content             user content: ""
  38_complex_tool_schema              tool with nested object / array / enum schema
  39_tool_choice_auto_no_parallel     tool_choice {type:"auto", disable_parallel_tool_use: true}
  40_tool_choice_none                 tool_choice {type:"none"}
  41_thinking_high                    thinking.budget_tokens = 12000 → reasoning_effort high
  42_thinking_low                     thinking.budget_tokens = 2000 → reasoning_effort low
  43_stop_sequences                   stop_sequences: ["END", "STOP"] → stop list
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "requests"


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


EXTRA = {
    "32_agent_tool_loop": {
        "model": "gpt-4o-mini",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_w1",
                        "name": "get_weather",
                        "input": {"city": "Tokyo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_w1", "content": "Sunny, 25C"}
                ],
            },
            {"role": "assistant", "content": "It's sunny and 25°C in Tokyo."},
        ],
    },
    "33_user_content_cache_control": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Very long cached context",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "Question: summarize."},
                ],
            }
        ],
    },
    "34_assistant_content_cache_control": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "ok"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "remembered answer",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "again"},
        ],
    },
    "35_assistant_thinking_history": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hard math problem"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me work this out step by step...", "signature": "sig_xyz"},
                    {"type": "text", "text": "The answer is 42."},
                ],
            },
            {"role": "user", "content": "Why?"},
        ],
    },
    "36_user_mixed_content": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_x", "content": "previous tool output"},
                    {"type": "text", "text": "Now look at this image:"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                ],
            }
        ],
    },
    "37_empty_string_content": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": ""}],
    },
    "38_complex_tool_schema": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "ok"}],
        "tools": [
            {
                "name": "search_flights",
                "description": "Find flights",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "passengers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "age": {"type": "integer", "minimum": 0},
                                    "class": {"type": "string", "enum": ["economy", "business", "first"]},
                                },
                                "required": ["age", "class"],
                            },
                        },
                        "departure_date": {"type": "string", "format": "date"},
                    },
                    "required": ["origin", "destination", "departure_date"],
                },
            }
        ],
    },
    "39_tool_choice_auto_no_parallel": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "ok"}],
        "tools": [{"name": "f", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    },
    "40_tool_choice_none": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "ok"}],
        "tools": [{"name": "f", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "none"},
    },
    "41_thinking_high": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 12000},
    },
    "42_thinking_low": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 2000},
    },
    "43_stop_sequences": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "tell a joke"}],
        "stop_sequences": ["END", "STOP"],
    },
}


def main() -> None:
    for name, payload in EXTRA.items():
        write(ROOT / f"anthropic_{name}.json", payload)
    print(f"wrote {len(EXTRA)} extra requests to {ROOT}")


if __name__ == "__main__":
    main()
