"""Python parity tests: feed each LiteLLM-golden fixture through the
PyO3-backed ``cc_convert.translate_request`` and assert the result matches
(semantically) the same LiteLLM golden the Rust test uses.

This validates that the Python wheel emits exactly what the Rust core does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cc_convert  # type: ignore

FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


def _normalize(v, ctx_key: str | None = None):
    """Drop nulls, recurse, parse-and-re-stringify tool_call `arguments`."""
    if v is None:
        return None
    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            n = _normalize(val, k)
            if n is None:
                continue
            out[k] = n
        return out
    if isinstance(v, list):
        return [_normalize(item) for item in v]
    if isinstance(v, str) and ctx_key == "arguments":
        try:
            return json.dumps(json.loads(v), separators=(",", ":"), sort_keys=True)
        except json.JSONDecodeError:
            return v
    return v


def _request_pairs():
    req_dir = FIXTURES / "requests"
    pairs = []
    for in_path in sorted(req_dir.glob("anthropic_*.json")):
        name = in_path.stem.removeprefix("anthropic_")
        golden = req_dir / f"openai_{name}.json"
        if golden.exists():
            pairs.append(pytest.param(in_path, golden, id=name))
    return pairs


@pytest.mark.parametrize("input_path,golden_path", _request_pairs())
def test_request_parity_with_litellm(input_path: Path, golden_path: Path) -> None:
    anthropic_req = json.loads(input_path.read_text())
    openai_actual, _tool_map = cc_convert.translate_request(
        anthropic_req, mode="litellm_compat"
    )
    golden = json.loads(golden_path.read_text())
    assert _normalize(openai_actual) == _normalize(golden), (
        f"Python wheel diverged from LiteLLM golden for {input_path.name}.\n"
        f"  actual: {json.dumps(_normalize(openai_actual), sort_keys=True, indent=2)}\n"
        f"  golden: {json.dumps(_normalize(golden), sort_keys=True, indent=2)}"
    )


def test_round_trip_response() -> None:
    """Trivial sanity check that translate_response works end-to-end."""
    out = cc_convert.translate_response(
        {
            "id": "chatcmpl-abc",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        },
        "claude-opus-4-7",
        {},
    )
    assert out["id"] == "msg_abc"
    assert out["content"][0]["text"] == "hi"
    assert out["stop_reason"] == "end_turn"


def test_stream_translator_text_only() -> None:
    t = cc_convert.StreamTranslator("claude-opus-4-7", {})
    events = []
    events += t.push({"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "hi"}}]})
    events += t.push(
        {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    kinds = [e["type"] for e in events]
    assert "message_start" in kinds
    assert "message_stop" in kinds
    # The text content was emitted as a delta.
    text_chunks = [
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"].get("type") == "text_delta"
    ]
    assert "hi" in "".join(text_chunks)


def test_pragmatic_collapses_single_text_system_to_string() -> None:
    """Real-world OAI upstreams (SGLang/vLLM strict mode) reject list-content
    on system messages. The 'pragmatic' default collapses single-text blocks
    to a plain string. The 'litellm_compat' mode keeps them as a list."""
    req = {
        "model": "Qwen",
        "max_tokens": 10,
        "system": [
            {"type": "text", "text": "rule A", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    pragmatic, _ = cc_convert.translate_request(req)  # default mode
    assert pragmatic["messages"][0]["role"] == "system"
    assert pragmatic["messages"][0]["content"] == "rule A", (
        "pragmatic should collapse single-text system to string"
    )

    litellm, _ = cc_convert.translate_request(req, mode="litellm_compat")
    assert litellm["messages"][0]["content"] == [{"type": "text", "text": "rule A"}], (
        "litellm_compat should keep list-content"
    )


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        cc_convert.translate_request({"model": "x", "max_tokens": 1, "messages": []}, mode="bogus")
