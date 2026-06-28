//! Cases 21–26 from the plan: response translation (OpenAI → Anthropic).

use cc_convert_core::anthropic::AnthropicStopReason;
use cc_convert_core::openai::OpenAIResponse;
use cc_convert_core::tool_names::ToolNameMap;
use cc_convert_core::openai_response_to_anthropic;
use serde_json::{json, Value};

fn convert(resp: Value, model: &str, map: Option<Value>) -> Value {
    let parsed: OpenAIResponse = serde_json::from_value(resp).expect("parse");
    let tm: ToolNameMap = match map {
        Some(v) => serde_json::from_value(v).unwrap(),
        None => ToolNameMap::new(),
    };
    let out = openai_response_to_anthropic(&parsed, model, &tm).expect("translate");
    serde_json::to_value(&out).unwrap()
}

#[test]
fn case21_plain_text_response() {
    let out = convert(
        json!({
            "id": "chatcmpl-abc",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello world"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4}
        }),
        "claude-opus-4-7",
        None,
    );
    assert_eq!(out["id"], "msg_abc");
    assert_eq!(out["model"], "claude-opus-4-7");
    assert_eq!(out["role"], "assistant");
    assert_eq!(out["type"], "message");
    assert_eq!(out["content"][0]["type"], "text");
    assert_eq!(out["content"][0]["text"], "hello world");
    assert_eq!(out["stop_reason"], "end_turn");
    assert_eq!(out["usage"]["input_tokens"], 10);
    assert_eq!(out["usage"]["output_tokens"], 4);
}

#[test]
fn case22_empty_content_emits_empty_text() {
    let out = convert(
        json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null},
                "finish_reason": "stop"
            }]
        }),
        "claude-opus-4-7",
        None,
    );
    assert_eq!(out["content"][0]["type"], "text");
    assert_eq!(out["content"][0]["text"], "");
}

#[test]
fn case23_single_tool_call_no_text() {
    let out = convert(
        json!({
            "id": "chatcmpl-x",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
        "claude-opus-4-7",
        None,
    );
    assert_eq!(out["content"].as_array().unwrap().len(), 1);
    assert_eq!(out["content"][0]["type"], "tool_use");
    assert_eq!(out["content"][0]["name"], "get_weather");
    assert_eq!(out["content"][0]["id"], "call_1");
    assert_eq!(out["content"][0]["input"]["city"], "Paris");
    assert_eq!(out["stop_reason"], "tool_use");
}

#[test]
fn case24_multiple_tool_calls() {
    let out = convert(
        json!({
            "id": "chatcmpl-x",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                        {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}}
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }),
        "claude-opus-4-7",
        None,
    );
    let blocks = out["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0]["name"], "f");
    assert_eq!(blocks[1]["name"], "g");
}

#[test]
fn case25_length_finish_maps_to_max_tokens() {
    let out = convert(
        json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "trunc"},
                "finish_reason": "length"
            }]
        }),
        "claude-opus-4-7",
        None,
    );
    assert_eq!(out["stop_reason"], "max_tokens");
}

#[test]
fn case26_cached_tokens_mapped() {
    let out = convert(
        json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 32}
            }
        }),
        "claude-opus-4-7",
        None,
    );
    assert_eq!(out["usage"]["cache_read_input_tokens"], 32);
}

#[test]
fn long_tool_name_round_trip_uses_map() {
    let long = "x".repeat(80);
    let mut m = ToolNameMap::new();
    let translated = m.translate(&long);
    let _ = AnthropicStopReason::EndTurn; // touch enum so it doesn't go unused

    let out = convert(
        json!({
            "id": "chatcmpl-x",
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": translated, "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
        "claude-opus-4-7",
        Some(serde_json::to_value(&m).unwrap()),
    );
    assert_eq!(out["content"][0]["name"], long);
}
