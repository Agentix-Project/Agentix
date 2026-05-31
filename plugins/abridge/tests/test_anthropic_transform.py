"""Tests for the Anthropic <-> OpenAI translation."""

from __future__ import annotations

import json

from agentix.bridge.transform import (
    anthropic_messages_to_openai,
    anthropic_sse,
    count_anthropic_tokens,
    openai_to_anthropic_messages,
)


def test_anthropic_system_string_lifted_into_first_message() -> None:
    body = {
        "model": "claude-3-haiku",
        "system": "You are helpful.",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    openai_body = anthropic_messages_to_openai(body, upstream_model="gpt-4o")
    assert openai_body["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert openai_body["messages"][1] == {"role": "user", "content": "Hello"}
    assert openai_body["model"] == "gpt-4o"
    assert openai_body["max_tokens"] == 100


def test_anthropic_system_block_list_joined() -> None:
    body = {
        "model": "claude",
        "system": [
            {"type": "text", "text": "be"},
            {"type": "text", "text": "concise"},
        ],
        "max_tokens": 50,
        "messages": [],
    }
    openai_body = anthropic_messages_to_openai(body)
    assert openai_body["messages"][0] == {"role": "system", "content": "be\nconcise"}


def test_anthropic_tool_use_lifted_to_openai_tool_call() -> None:
    body = {
        "model": "claude",
        "max_tokens": 100,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            },
        ],
    }
    openai_body = anthropic_messages_to_openai(body)
    assistant = openai_body["messages"][0]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"
    args = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
    assert args == {"command": "ls"}

    tool_msg = openai_body["messages"][1]
    assert tool_msg == {"role": "tool", "tool_call_id": "tool-1", "content": "ok"}


def test_anthropic_tools_to_openai_function_definitions() -> None:
    body = {
        "model": "claude",
        "max_tokens": 50,
        "messages": [],
        "tools": [
            {"name": "search", "description": "search the web", "input_schema": {"type": "object"}},
            # No name → skipped.
            {"description": "skip me"},
        ],
    }
    openai_body = anthropic_messages_to_openai(body)
    assert openai_body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search the web",
                "parameters": {"type": "object"},
            },
        }
    ]


def test_openai_response_to_anthropic_text_only() -> None:
    openai = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi there"},
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    anthropic = openai_to_anthropic_messages(openai, response_model="claude-3-haiku")
    assert anthropic["model"] == "claude-3-haiku"
    assert anthropic["stop_reason"] == "end_turn"
    assert anthropic["content"] == [{"type": "text", "text": "hi there"}]
    assert anthropic["usage"] == {"input_tokens": 4, "output_tokens": 2}


def test_openai_response_to_anthropic_tool_calls_lifted() -> None:
    openai = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    anthropic = openai_to_anthropic_messages(openai, response_model="claude")
    assert anthropic["stop_reason"] == "tool_use"
    assert anthropic["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "bash",
        "input": {"command": "ls"},
    }


def test_anthropic_sse_renders_text_and_tool_blocks() -> None:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    blob = anthropic_sse(body).decode()
    # Each event begins with `event: <name>\n` and ends with `\n\n`.
    assert "event: message_start" in blob
    assert "event: content_block_start" in blob
    assert "text_delta" in blob
    assert "input_json_delta" in blob
    expected_tail = (
        "event: message_stop\ndata: " + json.dumps({"type": "message_stop"}, separators=(",", ":")) + "\n\n"
    )
    assert blob.endswith(expected_tail)


def test_count_anthropic_tokens_smoke() -> None:
    body = {
        "system": "x" * 8,
        "messages": [
            {"role": "user", "content": "y" * 16},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "z" * 4},
                    {"type": "tool_result", "tool_use_id": "1", "content": "w" * 12},
                ],
            },
        ],
    }
    result = count_anthropic_tokens(body)
    # 8 + 16 + 4 + 12 = 40 chars; 40 // 4 = 10 estimated tokens.
    assert result.input_tokens == 10


def test_tool_loop_turn_ordering_matches_openai_protocol():
    """Assistant text+tool_use -> ONE message; the tool_result becomes a
    `tool` message immediately after it (OpenAI requires that adjacency).
    Regression: Claude Code's multi-turn tool loop broke without this.
    """
    body = {
        "model": "claude-x",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll run it"},
                    {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"cmd": "ls"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "file.txt"}]},
        ],
    }
    out = anthropic_messages_to_openai(body)["messages"]
    roles = [m["role"] for m in out]

    asst = [m for m in out if m["role"] == "assistant"]
    assert len(asst) == 1, f"expected one assistant message, got {roles}"
    assert asst[0]["content"] == "I'll run it"
    assert asst[0]["tool_calls"][0]["id"] == "tu_1"

    ai = roles.index("assistant")
    assert roles[ai + 1] == "tool", f"tool result must follow tool_calls: {roles}"
    assert out[ai + 1]["tool_call_id"] == "tu_1"
    assert "file.txt" in out[ai + 1]["content"]
