"""Golden-fixture parity tests for the Anthropic <-> OpenAI transforms.

Runs the pure converters in `agentix.bridge.clients._anthropic_transforms`
against the cc_convert conversion corpus vendored under
`fixtures/anthropic_openai/` (see the README there for origin, license,
and how the goldens were produced).

Three directions:

- requests:  `anthropic_<case>.json` -> `anthropic_messages_to_openai`
  -> compare to `openai_<case>.json`.
- responses: `openai_<case>.json` -> `openai_to_anthropic_messages`
  -> compare to `anthropic_<case>.json` (the generated message id is
  normalized away — see `_normalized_anthropic_response`).
- streams:   the Python path has no incremental OpenAI-SSE translator —
  the abridge client makes a *non-streaming* upstream call and replays
  the finished response as Anthropic SSE via `anthropic_sse`. Chunk-by-
  chunk event fidelity is therefore untestable; what IS testable is
  final-assembled-response equivalence: assemble the OpenAI chunks into
  a completed response (test helper), convert it, render it back out
  through `anthropic_sse`, re-assemble our own event stream, and compare
  that against the message assembled from the golden event stream.

Every known Python-path gap is marked `xfail(strict=True)` with a
precise reason, so the xfail table below doubles as the prioritized
gap list for the transforms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agentix.bridge.clients._anthropic_transforms import (
    anthropic_messages_to_openai,
    anthropic_sse,
    openai_to_anthropic_messages,
)

FIXTURES = Path(__file__).parent / "fixtures" / "anthropic_openai"

# ── known Python-path gaps (case -> xfail reason) ──────────────────────────

REQUEST_GAPS: dict[str, str] = {
    "05_user_image_base64": "base64 image blocks in user content are dropped (no image_url part is emitted)",
    "06_user_image_url": "a user message whose content is only an image block is dropped entirely",
    "11_user_tool_result_multipart": "image parts inside a multipart tool_result are dropped (only text is forwarded)",
    "13_long_tool_name": ">64-char tool names are not sanitized (no truncate+hash rename, no tool_map)",
    "14_tool_choice_any": "tool_choice {type: any} is dropped (should map to OpenAI tool_choice 'required')",
    "15_tool_choice_named": "tool_choice {type: tool, name} is dropped (should map to a named function tool_choice)",
    "16_metadata_user_id": "metadata.user_id is dropped (should map to the OpenAI 'user' field)",
    "17_thinking_medium": "thinking {budget_tokens: 5000} is dropped (no reasoning_effort='medium' mapping)",
    "18_top_k_dropped": "top_k is dropped; the oracle passes it through for OpenAI-compatible upstreams",
    "19_stream_include_usage": "stream=true is dropped (the adapter always makes a non-streaming upstream call)",
    "35_assistant_thinking_history": "assistant thinking blocks are dropped instead of forwarded as thinking_blocks",
    "36_user_mixed_content": "the image part of mixed text+image user content is dropped",
    "37_empty_string_content": "a message with empty-string content is forwarded; the oracle drops it",
    "39_tool_choice_auto_no_parallel": "tool_choice {type: auto} is dropped (should map to OpenAI tool_choice 'auto')",
    "40_tool_choice_none": "tool_choice {type: none} is dropped (should map to OpenAI tool_choice 'none')",
    "41_thinking_high": "thinking with a large budget is dropped (no reasoning_effort='high' mapping)",
    "42_thinking_low": "thinking {budget_tokens: 2000} is dropped (no reasoning_effort='low' mapping)",
    "43_stop_sequences": "stop_sequences is dropped (never mapped to an OpenAI stop parameter)",
}

RESPONSE_GAPS: dict[str, str] = {
    "22_empty_content": "an empty assistant message becomes [{type: text, text: ''}] instead of content: []",
    "26_cached_tokens": (
        "usage.prompt_tokens_details.cached_tokens is ignored: no cache_read_input_tokens, "
        "and input_tokens is not reduced by the cached share"
    ),
}

STREAM_GAPS: dict[str, str] = {
    "30_stream_ends_without_finish_reason": (
        "a stream aborted without finish_reason is reported as stop_reason 'end_turn'; "
        "the oracle keeps stop_reason null and emits no message_delta"
    ),
    "31_reasoning_then_text": "reasoning_content deltas are dropped; no thinking block is produced",
}

# Cases where the golden itself is not a valid oracle (upstream artifact,
# not a Python gap) — skipped, not xfailed, so the xfail list stays a pure
# gap list.
STREAM_SKIPS: dict[str, str] = {
    "29_two_parallel_tool_calls": (
        "the golden merges both parallel tool calls into ONE content block with concatenated "
        "partial_json '{}{}' (LiteLLM stream-translator artifact); the Python one-shot path emits "
        "two separate tool_use blocks, so there is no valid oracle to compare against"
    ),
}


def _cases(directory: str, pattern: str, gaps: dict[str, str], skips: dict[str, str] | None = None) -> list[Any]:
    params: list[Any] = []
    for path in sorted((FIXTURES / directory).glob(pattern)):
        case = path.stem.split("_", 1)[1]
        marks: list[pytest.MarkDecorator] = []
        if case in gaps:
            marks.append(pytest.mark.xfail(strict=True, reason=gaps[case]))
        if skips and case in skips:
            marks.append(pytest.mark.skip(reason=skips[case]))
        params.append(pytest.param(case, marks=marks, id=case))
    return params


def _load(directory: str, name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / directory / name).read_text())


# ── comparison normalization ────────────────────────────────────────────────


def _drop_none(value: Any) -> Any:
    """Recursively drop dict entries whose value is None.

    JSON `null` and an absent key are interchangeable on both wire
    protocols, and the goldens carry LiteLLM serialization artifacts
    that spell absence as an explicit null (`"thinking_blocks": null`,
    `"provider_specific_fields": null`, `"content": null`). This only
    equates null with absence — a null where a real value is expected
    (or vice versa) still fails.
    """
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def _canonical_message_content(content: Any) -> Any:
    """Fold a text-only content part list into a plain string.

    OpenAI treats `content: "s"` and `content: [{"type": "text", ...}]`
    as equivalent. The goldens keep Anthropic's block boundaries (those
    only carry cache_control, which has no OpenAI equivalent) while the
    Python transform joins text blocks with a newline — so both sides
    are folded with the same newline join. The fold applies ONLY when
    every part is a text part: any non-text part (e.g. image_url) keeps
    the list shape, so dropped images still fail loudly.
    """
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" for part in content):
        return "\n".join(str(part.get("text", "")) for part in content)
    return content


def _normalized_openai_request(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = _drop_none(body)
    for message in out.get("messages", []):
        if "content" in message:
            message["content"] = _canonical_message_content(message["content"])
    for tool in out.get("tools", []):
        function = tool.get("function", {})
        # The transform always emits `description` ("" when the Anthropic
        # tool declares none); the golden omits the key entirely. An empty
        # description carries no information, so "" == absent here.
        if function.get("description") == "":
            del function["description"]
    return out


def _normalized_anthropic_response(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = _drop_none(body)
    # The transform mints a fresh `msg_<uuid>` message id while the golden
    # echoes the upstream OpenAI id — both are generated values, so the id
    # (and only the id) is excluded from the comparison.
    out.pop("id", None)
    return out


# ── requests: Anthropic -> OpenAI ───────────────────────────────────────────


@pytest.mark.parametrize("case", _cases("requests", "anthropic_*.json", REQUEST_GAPS))
def test_request_transform_matches_golden(case: str) -> None:
    anthropic_body = _load("requests", f"anthropic_{case}.json")
    expected = _load("requests", f"openai_{case}.json")

    actual = anthropic_messages_to_openai(anthropic_body)

    assert _normalized_openai_request(actual) == _normalized_openai_request(expected)


# ── responses: OpenAI -> Anthropic ──────────────────────────────────────────


@pytest.mark.parametrize("case", _cases("responses", "openai_*.json", RESPONSE_GAPS))
def test_response_transform_matches_golden(case: str) -> None:
    openai_body = _load("responses", f"openai_{case}.json")
    expected = _load("responses", f"anthropic_{case}.json")
    meta = _load("responses", f"meta_{case}.json")
    # The Python path has no tool_map (reverse name-mapping) support; every
    # golden in the corpus was produced with an empty map, so nothing is
    # being papered over. A future non-empty map must not pass silently.
    assert meta["tool_map"] == {}

    # `response_model` is caller-supplied; the goldens echo the upstream
    # OpenAI model (meta.original_model is not used by the reference
    # translator's output), so pass the upstream model through.
    actual = openai_to_anthropic_messages(openai_body, response_model=openai_body["model"])

    assert _normalized_anthropic_response(actual) == _normalized_anthropic_response(expected)


# ── streams: final assembled response equivalence ───────────────────────────


def _parse_openai_sse(text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


def _assemble_openai_stream(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge Chat Completions chunks into one completed response.

    Test-side helper standing in for the non-streaming upstream call the
    adapter actually makes: content and per-tool-call argument fragments
    are concatenated, tool calls are keyed by their delta `index`, and the
    last non-null finish_reason wins. `reasoning_content` fragments are
    preserved so the reasoning gap fails on the transform, not on the
    assembly.
    """
    response_id: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}

    for chunk in chunks:
        response_id = chunk.get("id") or response_id
        model = chunk.get("model") or model
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            for fragment in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    fragment.get("index", 0),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if fragment.get("id"):
                    slot["id"] = fragment["id"]
                function = fragment.get("function") or {}
                if function.get("name"):
                    slot["function"]["name"] = function["name"]
                if function.get("arguments"):
                    slot["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

    completed: dict[str, Any] = {
        "id": response_id,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if usage:
        completed["usage"] = usage
    return completed


def _parse_anthropic_sse(blob: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in blob.decode().splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _assemble_anthropic_stream(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble an Anthropic event stream into its final message shape.

    Returns `{"content": [...], "stop_reason": ...}` — the semantic core
    of the stream. Ids, model, and usage are deliberately excluded: the
    goldens use a freshly generated `msg_<uuid>` id, the caller-level
    `original_model`, and all-zero usage, none of which the body-level
    transforms control. Two normalizations:

    - empty text blocks are dropped (the golden stream translator opens
      an unconditional empty text block before every message — a LiteLLM
      artifact carrying no content);
    - `thinking_delta` fragments become a separate thinking part even
      when the golden interleaves them inside a text block, so thinking
      content is compared by value, not by block layout.
    """
    blocks: dict[int, dict[str, Any]] = {}
    stop_reason: str | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "content_block_start":
            start = dict(event.get("content_block") or {})
            blocks[event["index"]] = {
                "type": start.get("type", "text"),
                "id": start.get("id"),
                "name": start.get("name"),
                "text": start.get("text", ""),
                "thinking": start.get("thinking", ""),
                "partial_json": "",
            }
        elif event_type == "content_block_delta":
            block = blocks.setdefault(
                event["index"],
                {"type": "text", "id": None, "name": None, "text": "", "thinking": "", "partial_json": ""},
            )
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] += delta.get("text", "")
            elif delta_type == "thinking_delta":
                block["thinking"] += delta.get("thinking", "")
            elif delta_type == "input_json_delta":
                block["partial_json"] += delta.get("partial_json", "")
        elif event_type == "message_delta":
            delta = event.get("delta") or {}
            if delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]

    content: list[dict[str, Any]] = []
    for index in sorted(blocks):
        block = blocks[index]
        if block["thinking"]:
            content.append({"type": "thinking", "thinking": block["thinking"]})
        if block["type"] == "tool_use":
            raw = block["partial_json"] or "{}"
            try:
                tool_input: Any = json.loads(raw)
            except json.JSONDecodeError:
                tool_input = raw
            content.append({"type": "tool_use", "id": block["id"], "name": block["name"], "input": tool_input})
        elif block["text"]:
            content.append({"type": "text", "text": block["text"]})
    return {"content": content, "stop_reason": stop_reason}


@pytest.mark.parametrize("case", _cases("streams", "openai_*.sse", STREAM_GAPS, STREAM_SKIPS))
def test_stream_final_assembly_matches_golden(case: str) -> None:
    chunks = _parse_openai_sse((FIXTURES / "streams" / f"openai_{case}.sse").read_text())
    golden_events = [
        json.loads(line)
        for line in (FIXTURES / "streams" / f"anthropic_{case}.jsonl").read_text().splitlines()
        if line.strip()
    ]

    completed = _assemble_openai_stream(chunks)
    # The stream goldens carry the original (Anthropic-side) model name at
    # message_start; it is caller-supplied, exactly like `response_model`.
    body = openai_to_anthropic_messages(completed, response_model="claude-opus-4-7")
    actual_events = _parse_anthropic_sse(anthropic_sse(body))

    assert _assemble_anthropic_stream(actual_events) == _assemble_anthropic_stream(golden_events)


# ── corpus integrity ────────────────────────────────────────────────────────


def test_fixture_corpus_is_complete() -> None:
    """Guard against a botched fixture copy: every case must be a full
    triple/pair, and the corpus must have the expected size."""
    request_cases = {p.stem.split("_", 1)[1] for p in (FIXTURES / "requests").glob("anthropic_*.json")}
    assert len(request_cases) == 32
    for case in request_cases:
        assert (FIXTURES / "requests" / f"openai_{case}.json").is_file()
        assert (FIXTURES / "requests" / f"tool_map_{case}.json").is_file()

    response_cases = {p.stem.split("_", 1)[1] for p in (FIXTURES / "responses").glob("openai_*.json")}
    assert len(response_cases) == 6
    for case in response_cases:
        assert (FIXTURES / "responses" / f"anthropic_{case}.json").is_file()
        assert (FIXTURES / "responses" / f"meta_{case}.json").is_file()

    stream_cases = {p.stem.split("_", 1)[1] for p in (FIXTURES / "streams").glob("openai_*.sse")}
    assert len(stream_cases) == 5
    for case in stream_cases:
        assert (FIXTURES / "streams" / f"anthropic_{case}.jsonl").is_file()
