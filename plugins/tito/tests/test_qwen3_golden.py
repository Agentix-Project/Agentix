"""Golden test: the incremental==from-scratch invariant on the REAL Qwen3
tokenizer + the bundled fixed chat template.

Every other engine test runs on a tiny in-memory WordLevel tokenizer, which
cannot catch real-tokenizer failure modes (BPE merge boundaries at segment
junctions, `<think>`/tool-tag added-token splitting, the missing trailing
newline after `<|im_end|>`). This module downloads the tokenizer-only files
for Qwen/Qwen3-0.6B at test time (a few hundred KB; the HF cache is reused on
later runs) and drives a full multi-turn tool-calling session through
`LinearTrajectory.prepare_prompt`, asserting:

- each turn's incrementally merged prompt ids equal a from-scratch render of
  the same request (token-exact, not just segment-equivalent);
- the accumulated trajectory equals the from-scratch render of the whole
  conversation (modulo the trailing newline Qwen3 omits at stop);
- the read-time mismatch audit reports clean;
- the `prompt_segments` spans the per-turn record persists decode to the
  expected role boundaries.

Offline behavior: if the tokenizer is neither cached nor downloadable the
module SKIPS (marker: `network`) — it never fails a disconnected run.
"""

from __future__ import annotations

import json

import pytest
from agentix.tito.engine.pretokenize import get_tito_tokenizer
from agentix.tito.engine.trajectory import LinearTrajectory, SessionRecord, SessionRegistry

pytestmark = pytest.mark.network

_REPO = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen3_tok():
    from transformers import AutoTokenizer

    try:  # cache first: offline runs with a warm cache still exercise the golden
        return AutoTokenizer.from_pretrained(_REPO, local_files_only=True)
    except Exception:
        pass
    try:  # one download attempt; a blocked network is a skip, not a failure
        return AutoTokenizer.from_pretrained(_REPO)
    except Exception as exc:  # noqa: BLE001 - hub errors vary by transport
        pytest.skip(f"Qwen3 tokenizer unavailable (offline?): {type(exc).__name__}: {exc}")


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _simulate_completion(tt, request_messages, assistant_message, prompt_ids, tools):
    """The completion token ids a template-canonical model would emit for
    *assistant_message*: the from-scratch render of request+assistant minus
    the prompt prefix, with the trailing newline dropped (Qwen3 stops at
    `<|im_end|>` and never emits the newline the template writes)."""
    full = tt.render_messages(
        request_messages + [assistant_message], tools=tools, add_generation_prompt=False, tokenize=True
    )
    assert full[: len(prompt_ids)] == prompt_ids, "assistant render must extend the generation prompt"
    completion = full[len(prompt_ids):]
    newline_id = tt.tokenizer.encode("\n", add_special_tokens=False)[0]
    assert completion and completion[-1] == newline_id
    return completion[:-1]


def test_qwen3_incremental_equals_from_scratch_multi_turn_tool_calls(qwen3_tok):
    tt = get_tito_tokenizer(qwen3_tok, "qwen3", allowed_append_roles=("tool", "user"))
    registry = SessionRegistry(None, qwen3_tok, tito_tokenizer=tt)
    tr = LinearTrajectory()

    system = {"role": "system", "content": "You are a terse weather assistant."}
    user1 = {"role": "user", "content": "What's the weather in Paris right now?"}
    assistant1 = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "The user wants current weather; call the tool.",
        "tool_calls": [
            {
                "id": "call_0001",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
            }
        ],
    }
    tool1 = {"role": "tool", "content": '{"temp_c": 21, "sky": "clear"}', "tool_call_id": "call_0001"}
    assistant2 = {
        "role": "assistant",
        "content": "Paris is 21°C and clear.",
        "reasoning_content": "Tool says 21C, clear. Answer briefly.",
    }
    user2 = {"role": "user", "content": "And in London?"}
    assistant3 = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Same tool, city London.",
        "tool_calls": [
            {
                "id": "call_0002",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "London"})},
            }
        ],
    }

    turns = [
        ([system, user1], assistant1),
        ([system, user1, assistant1, tool1], assistant2),
        ([system, user1, assistant1, tool1, assistant2, user2], assistant3),
    ]

    all_segment_sources: list[list[str]] = []
    for request_messages, assistant in turns:
        prepared = tr.prepare_prompt(request_messages, _TOOLS, tito_tokenizer=tt)

        # THE invariant, token-exact on the real tokenizer: the incrementally
        # merged prompt equals a from-scratch render of the same request.
        from_scratch = tt.render_messages(
            request_messages, tools=_TOOLS, add_generation_prompt=True, tokenize=True
        )
        assert prepared.token_ids == from_scratch
        assert prepared.prefix_stable is True

        # Segment spans tile the prompt exactly.
        assert prepared.segments[0]["start"] == 0
        assert prepared.segments[-1]["end"] == len(prepared.token_ids)
        for left, right in zip(prepared.segments, prepared.segments[1:], strict=False):
            assert left["end"] == right["start"]
        all_segment_sources.append([s["source"] for s in prepared.segments])

        completion = _simulate_completion(tt, request_messages, assistant, prepared.token_ids, _TOOLS)
        tr.update_pretokenized_state(
            request_messages,
            assistant,
            prompt_token_ids=prepared.token_ids,
            completion_token_ids=completion,
            max_trim_tokens=tt.max_trim_tokens,
        )
        # As the gateway does — the audit reads tools off the last record.
        tr.append_record(SessionRecord(
            timestamp=0.0, method="POST", path="/v1/chat/completions", status_code=200,
            request={"model": "m", "messages": request_messages, "tools": _TOOLS}, response={},
        ))

    assert all_segment_sources == [
        ["render"],
        ["prefix", "tool", "generation_prompt"],
        ["prefix", "user", "generation_prompt"],
    ]

    # Accumulated trajectory == from-scratch render of the full conversation
    # (fix_prefix restores the trailing newline Qwen3 omitted at stop).
    final_messages = turns[-1][0] + [assistant3]
    assert tt.fix_prefix(tr.token_ids) == tt.render_messages(
        final_messages, tools=_TOOLS, add_generation_prompt=False, tokenize=True
    )

    # The read-time audit agrees: no structural or content mismatch.
    assert registry.compute_session_mismatch(tr) == []


def test_qwen3_prompt_segment_boundaries_decode_to_role_markers(qwen3_tok):
    """The spans persisted as record.prompt_segments decode to the expected
    template boundaries — the loss-mask construction material downstream."""
    tt = get_tito_tokenizer(qwen3_tok, "qwen3", allowed_append_roles=("tool", "user"))
    tr = LinearTrajectory()

    request1 = [
        {"role": "system", "content": "You are a terse weather assistant."},
        {"role": "user", "content": "What's the weather in Paris right now?"},
    ]
    assistant1 = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Call the tool.",
        "tool_calls": [
            {
                "id": "call_0001",
                "type": "function",
                "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})},
            }
        ],
    }
    prepared1 = tr.prepare_prompt(request1, _TOOLS, tito_tokenizer=tt)
    completion1 = _simulate_completion(tt, request1, assistant1, prepared1.token_ids, _TOOLS)
    tr.update_pretokenized_state(
        request1, assistant1,
        prompt_token_ids=prepared1.token_ids,
        completion_token_ids=completion1,
        max_trim_tokens=tt.max_trim_tokens,
    )

    request2 = request1 + [assistant1, {"role": "tool", "content": "21C clear", "tool_call_id": "call_0001"}]
    prepared2 = tr.prepare_prompt(request2, _TOOLS, tito_tokenizer=tt)

    def decode(segment):
        return qwen3_tok.decode(
            prepared2.token_ids[segment["start"]:segment["end"]], skip_special_tokens=False
        )

    prefix_seg, tool_seg, gen_seg = prepared2.segments
    assert prefix_seg["source"] == "prefix"
    # fix_prefix restored the newline after the completion's final <|im_end|>.
    assert decode(prefix_seg).endswith("<|im_end|>\n")
    assert "<tool_call>" in decode(prefix_seg)  # the assistant turn lives in the prefix
    assert tool_seg["source"] == "tool"
    assert "<tool_response>" in decode(tool_seg) and "21C clear" in decode(tool_seg)
    assert gen_seg["source"] == "generation_prompt"
    assert decode(gen_seg) == "<|im_start|>assistant\n"
