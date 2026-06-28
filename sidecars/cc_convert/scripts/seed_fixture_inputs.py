"""Seed all 31 golden-fixture inputs.

Run once: `python scripts/seed_fixture_inputs.py`. Writes:

  tests/fixtures/requests/anthropic_<name>.json     (cases 1-20)
  tests/fixtures/responses/openai_<name>.json       (cases 21-26)
  tests/fixtures/responses/meta_<name>.json         (original_model + tool_map)
  tests/fixtures/streams/openai_<name>.sse          (cases 27-31)

The corresponding `openai_*.json` / `anthropic_*.json` / `anthropic_*.jsonl`
golden outputs are produced by `scripts/regen_fixtures.py` (which calls
LiteLLM) or by `scripts/seed_golden_outputs.py` (which calls the Rust
translator we just built — useful when LiteLLM is unreachable).
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text)


# --------- Requests ---------

REQUESTS = {
    "01_plain_user_text": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
    },
    "02_system_string": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "system": "Be concise.",
        "messages": [{"role": "user", "content": "hi"}],
    },
    "03_system_blocks_with_cache_control": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "rule 1", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": "hi"}],
    },
    "04_multi_turn": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "ok"},
        ],
    },
    "05_user_image_base64": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                    },
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
    },
    "06_user_image_url": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
                ],
            }
        ],
    },
    "07_assistant_single_tool_use": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                ],
            }
        ],
    },
    "08_assistant_two_parallel_tool_uses": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_a", "name": "f1", "input": {"a": 1}},
                    {"type": "tool_use", "id": "tu_b", "name": "f2", "input": {"b": 2}},
                ],
            }
        ],
    },
    "09_user_single_tool_result": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "21C"}
                ],
            }
        ],
    },
    "10_user_three_tool_results": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
                    {"type": "tool_result", "tool_use_id": "t3", "content": "c"},
                ],
            }
        ],
    },
    "11_user_tool_result_multipart": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "see image:"},
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "https://x/y.png"},
                            },
                        ],
                    }
                ],
            }
        ],
    },
    "12_tools_input_schema": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "weather lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    },
    "13_long_tool_name": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "x" * 80, "input_schema": {"type": "object"}}],
    },
    "14_tool_choice_any": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "any"},
    },
    "15_tool_choice_named": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "f", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "f"},
    },
    "16_metadata_user_id": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": "u-123"},
    },
    "17_thinking_medium": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 5000},
    },
    "18_top_k_dropped": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 20,
    },
    "19_stream_include_usage": {
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    },
    "20_o3_mini_uses_max_completion_tokens": {
        "model": "o3-mini",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "hi"}],
    },
}


# --------- Responses ---------

RESPONSES = {
    "21_plain_text": (
        {
            "id": "chatcmpl-abc",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
        {"original_model": "claude-opus-4-7", "tool_map": {}},
    ),
    "22_empty_content": (
        {
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
        },
        {"original_model": "claude-opus-4-7", "tool_map": {}},
    ),
    "23_single_tool_call_no_text": (
        {
            "id": "chatcmpl-x",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"original_model": "claude-opus-4-7", "tool_map": {}},
    ),
    "24_multiple_tool_calls": (
        {
            "id": "chatcmpl-y",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            },
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "g", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {"original_model": "claude-opus-4-7", "tool_map": {}},
    ),
    "25_length_max_tokens": (
        {
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "trunc"},
                    "finish_reason": "length",
                }
            ],
        },
        {"original_model": "claude-opus-4-7", "tool_map": {}},
    ),
    "26_cached_tokens": (
        {
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 32},
            },
        },
        {"original_model": "claude-opus-4-7", "tool_map": {}},
    ),
}


# --------- Streams (raw OpenAI SSE input) ---------

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


STREAMS_INPUT = {
    "27_text_only": (
        sse(
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hel"}}],
            }
        )
        + sse({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "lo"}}]})
        + sse(
            {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )
        + "data: [DONE]\n\n"
    ),
    "28_single_tool_call_fragments": (
        sse(
            {
                "id": "chatcmpl-x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                    }
                ],
            }
        )
        + sse(
            {
                "id": "chatcmpl-x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"arguments": '{"city":'},
                                }
                            ]
                        },
                    }
                ],
            }
        )
        + sse(
            {
                "id": "chatcmpl-x",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"arguments": '"Paris"}'},
                                }
                            ]
                        },
                    }
                ],
            }
        )
        + sse(
            {
                "id": "chatcmpl-x",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
    ),
    "29_two_parallel_tool_calls": (
        sse(
            {
                "id": "chatcmpl-y",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "a",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{}"},
                                },
                                {
                                    "index": 1,
                                    "id": "b",
                                    "type": "function",
                                    "function": {"name": "g", "arguments": "{}"},
                                },
                            ]
                        },
                    }
                ],
            }
        )
        + sse(
            {
                "id": "chatcmpl-y",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        )
    ),
    "30_stream_ends_without_finish_reason": (
        sse(
            {
                "id": "chatcmpl-z",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}}],
            }
        )
    ),
    "31_reasoning_then_text": (
        sse(
            {
                "id": "chatcmpl-r",
                "choices": [{"index": 0, "delta": {"reasoning_content": "let me think..."}}],
            }
        )
        + sse({"id": "chatcmpl-r", "choices": [{"index": 0, "delta": {"content": "Done."}}]})
        + sse(
            {"id": "chatcmpl-r", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )
    ),
}


def main() -> None:
    for name, payload in REQUESTS.items():
        write(ROOT / "requests" / f"anthropic_{name}.json", payload)
    for name, (payload, meta) in RESPONSES.items():
        write(ROOT / "responses" / f"openai_{name}.json", payload)
        write(ROOT / "responses" / f"meta_{name}.json", meta)
    for name, sse_text in STREAMS_INPUT.items():
        path = ROOT / "streams" / f"openai_{name}.sse"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sse_text)

    print(
        f"wrote {len(REQUESTS)} requests, {len(RESPONSES)} responses, "
        f"{len(STREAMS_INPUT)} streams to {ROOT}"
    )


if __name__ == "__main__":
    main()
