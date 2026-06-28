//! Cases 27–31 from the plan: streaming translation (OpenAI SSE → Anthropic SSE).

use cc_convert_core::anthropic::{AnthropicEvent, BlockDelta, StreamingContentBlock};
use cc_convert_core::openai::OpenAIStreamChunk;
use cc_convert_core::tool_names::ToolNameMap;
use cc_convert_core::{StreamConvertOptions, StreamTranslator};
use serde_json::{json, Value};

fn chunk(v: Value) -> OpenAIStreamChunk {
    serde_json::from_value(v).expect("parse chunk")
}

fn make() -> StreamTranslator {
    // Use anthropic-native preset so ping + lazy block opening + stop_sequence
    // assumptions hold for these unit tests. The litellm_compat parity test
    // is separate.
    StreamTranslator::with_options(
        "claude-opus-4-7".to_string(),
        ToolNameMap::new(),
        StreamConvertOptions::anthropic_native(),
    )
}

#[test]
fn case27_text_only_stream() {
    let mut t = make();
    let mut all = Vec::new();
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hel"}}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {"content": "lo"}}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }))));

    // Expect: message_start, ping, content_block_start(text), 2 deltas, content_block_stop,
    // message_delta, message_stop.
    assert!(matches!(all[0], AnthropicEvent::MessageStart { .. }));
    assert!(matches!(all[1], AnthropicEvent::Ping));
    assert!(matches!(all[2], AnthropicEvent::ContentBlockStart { ref content_block, .. } if matches!(content_block, StreamingContentBlock::Text { .. })));
    let mut text_seen = String::new();
    for ev in &all {
        if let AnthropicEvent::ContentBlockDelta { delta: BlockDelta::TextDelta { text }, .. } = ev {
            text_seen.push_str(text);
        }
    }
    assert_eq!(text_seen, "hello");
    assert!(matches!(all[all.len() - 1], AnthropicEvent::MessageStop));
}

#[test]
fn case28_single_tool_call_stream_with_fragments() {
    let mut t = make();
    let mut all = Vec::new();
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-x",
        "choices": [{"index": 0, "delta": {
            "tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": ""}
            }]
        }}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-x",
        "choices": [{"index": 0, "delta": {
            "tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"arguments": "{\"city\":"}
            }]
        }}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-x",
        "choices": [{"index": 0, "delta": {
            "tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"arguments": "\"Paris\"}"}
            }]
        }}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-x",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    }))));

    // First ContentBlockStart should be a tool_use named get_weather.
    let starts: Vec<_> = all
        .iter()
        .filter_map(|e| match e {
            AnthropicEvent::ContentBlockStart { content_block, index } => Some((index, content_block)),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 1);
    assert!(matches!(starts[0].1, StreamingContentBlock::ToolUse { name, .. } if name == "get_weather"));

    // Accumulated input_json_deltas should reconstruct the JSON.
    let mut buf = String::new();
    for ev in &all {
        if let AnthropicEvent::ContentBlockDelta { delta: BlockDelta::InputJsonDelta { partial_json }, .. } = ev {
            buf.push_str(partial_json);
        }
    }
    assert_eq!(buf, "{\"city\":\"Paris\"}");

    // Final stop_reason must be tool_use.
    let msg_delta = all.iter().find_map(|e| match e {
        AnthropicEvent::MessageDelta { delta, .. } => Some(delta),
        _ => None,
    }).unwrap();
    assert_eq!(msg_delta.stop_reason, Some(cc_convert_core::anthropic::AnthropicStopReason::ToolUse));
}

#[test]
fn case29_two_parallel_tool_calls_get_distinct_indices() {
    let mut t = make();
    let mut all = Vec::new();
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-y",
        "choices": [{"index": 0, "delta": {
            "tool_calls": [
                {"index": 0, "id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"index": 1, "id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}}
            ]
        }}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-y",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    }))));

    let mut indices: Vec<i32> = all.iter().filter_map(|e| match e {
        AnthropicEvent::ContentBlockStart { index, .. } => Some(*index),
        _ => None,
    }).collect();
    indices.sort();
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn case30_stream_ends_without_finish_reason() {
    let mut t = make();
    let mut all = Vec::new();
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-z",
        "choices": [{"index": 0, "delta": {"content": "partial"}}]
    }))));
    // No finish_reason chunk arrives — caller calls finish().
    all.extend(t.finish());

    // We must still see content_block_stop, message_delta (end_turn), message_stop.
    let has_stop = all.iter().any(|e| matches!(e, AnthropicEvent::MessageStop));
    let stop_reason_end_turn = all.iter().any(|e| matches!(e,
        AnthropicEvent::MessageDelta { delta, .. } if delta.stop_reason == Some(cc_convert_core::anthropic::AnthropicStopReason::EndTurn)
    ));
    assert!(has_stop);
    assert!(stop_reason_end_turn);
}

#[test]
fn case31_reasoning_content_emits_thinking_block() {
    let mut t = make();
    let mut all = Vec::new();
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-r",
        "choices": [{"index": 0, "delta": {"reasoning_content": "let me think..."}}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-r",
        "choices": [{"index": 0, "delta": {"content": "Done."}}]
    }))));
    all.extend(t.push_openai_chunk(&chunk(json!({
        "id": "chatcmpl-r",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }))));

    // First ContentBlockStart must be Thinking; second must be Text.
    let starts: Vec<_> = all.iter().filter_map(|e| match e {
        AnthropicEvent::ContentBlockStart { content_block, .. } => Some(content_block),
        _ => None,
    }).collect();
    assert!(matches!(starts[0], StreamingContentBlock::Thinking { .. }));
    assert!(matches!(starts[1], StreamingContentBlock::Text { .. }));
}
