//! vLLM- and SGLang-specific quirks. These tests verify that the translator
//! handles the non-standard fields and shapes these self-hosted
//! OpenAI-compatible servers emit, without panicking and producing
//! sensible Anthropic-side output.
//!
//! Source-grounded against:
//!   - vllm-project/vllm `vllm/entrypoints/openai/chat_completion/protocol.py`
//!     (`reasoning` field, `stop_reason` field, `routed_experts`)
//!   - sgl-project/sglang `python/sglang/srt/entrypoints/openai/protocol.py`
//!     and `serving_chat.py` (`reasoning_content` null-everywhere,
//!     null id/name on continuation tool_call chunks, `matched_stop`,
//!     `finish_reason: "abort"`)

use cc_convert_core::anthropic::{
    AnthropicEvent, AnthropicStopReason, BlockDelta, StreamingContentBlock,
};
use cc_convert_core::openai::{OpenAIResponse, OpenAIStreamChunk};
use cc_convert_core::resp_to_anthropic::openai_response_to_anthropic;
use cc_convert_core::tool_names::ToolNameMap;
use cc_convert_core::{StreamConvertOptions, StreamTranslator};
use serde_json::{json, Value};

fn translate_response(raw: Value) -> Value {
    let resp: OpenAIResponse = serde_json::from_value(raw).expect("parse OpenAIResponse");
    let out = openai_response_to_anthropic(&resp, "claude-opus-4-7", &ToolNameMap::new())
        .expect("translate");
    serde_json::to_value(&out).unwrap()
}

fn translator_native() -> StreamTranslator {
    StreamTranslator::with_options(
        "claude-opus-4-7".to_string(),
        ToolNameMap::new(),
        StreamConvertOptions::anthropic_native(),
    )
}

fn push(t: &mut StreamTranslator, raw: Value) -> Vec<AnthropicEvent> {
    let chunk: OpenAIStreamChunk = serde_json::from_value(raw).expect("parse chunk");
    t.push_openai_chunk(&chunk)
}

// ---------- vLLM ----------

#[test]
fn vllm_response_with_stop_reason_and_extra_fields_does_not_panic() {
    // vLLM emits stop_reason, prompt_logprobs, prompt_token_ids alongside
    // the standard fields. Our deserializer must ignore them gracefully.
    let raw = json!({
        "id": "chatcmpl-abc",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "object": "chat.completion",
        "prompt_logprobs": null,
        "prompt_token_ids": [1, 2, 3],
        "prompt_text": "hi",
        "kv_transfer_params": null,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello back",
                "reasoning": null,
                "tool_calls": []
            },
            "finish_reason": "stop",
            "stop_reason": "<|im_end|>",
            "token_ids": null,
            "routed_experts": null
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    });
    let out = translate_response(raw);
    assert_eq!(out["content"][0]["type"], "text");
    assert_eq!(out["content"][0]["text"], "hello back");
    assert_eq!(out["stop_reason"], "end_turn");
}

#[test]
fn vllm_response_with_reasoning_field_extracted_as_thinking() {
    // vLLM uses `reasoning`, not `reasoning_content`. We support both as
    // aliases in our deserializer (see openai.rs OpenAIChoiceMessage).
    let raw = json!({
        "id": "chatcmpl-x",
        "model": "deepseek-r1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning": "Let me think about the problem..."
            },
            "finish_reason": "stop"
        }]
    });
    let out = translate_response(raw);
    let content = out["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "thinking");
    assert_eq!(content[0]["thinking"], "Let me think about the problem...");
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "The answer is 42.");
}

#[test]
fn vllm_stream_reasoning_delta_via_reasoning_field() {
    let mut t = translator_native();
    let mut all = Vec::new();
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-r",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "reasoning": "Let me think"}
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-r",
            "choices": [{"index": 0, "delta": {"reasoning": " harder"}}]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-r",
            "choices": [{"index": 0, "delta": {"content": "42"}}]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-r",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }),
    ));

    // The thinking content should be reconstructed from the `reasoning` deltas.
    let mut buf = String::new();
    for ev in &all {
        if let AnthropicEvent::ContentBlockDelta {
            delta: BlockDelta::ThinkingDelta { thinking },
            ..
        } = ev
        {
            buf.push_str(thinking);
        }
    }
    assert_eq!(buf, "Let me think harder");
}

#[test]
fn vllm_initial_role_only_chunk_does_not_open_text_block() {
    // vLLM always emits a role+empty-content chunk first. Our translator
    // should not open a text content_block until real text arrives.
    let mut t = translator_native();
    let mut all = Vec::new();
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-v",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]
        }),
    ));
    // No text block start yet.
    let starts: Vec<_> = all
        .iter()
        .filter(|e| matches!(e, AnthropicEvent::ContentBlockStart { .. }))
        .collect();
    assert!(starts.is_empty(), "should not open a block on empty content");
    // Now real text arrives.
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-v",
            "choices": [{"index": 0, "delta": {"content": "hi"}}]
        }),
    ));
    let opened_text = all.iter().any(|e| matches!(e,
        AnthropicEvent::ContentBlockStart {
            content_block: StreamingContentBlock::Text { .. }, ..
        }));
    assert!(opened_text);
}

#[test]
fn vllm_tool_call_id_only_on_first_chunk() {
    // vLLM (and SGLang) emit `id` only on the first chunk for a tool_call.
    // Continuation chunks have just `index` + `function.arguments`.
    let mut t = translator_native();
    let mut all = Vec::new();
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-z",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0, "id": "chatcmpl-tool-abc", "type": "function",
                        "function": {"name": "search"}
                    }]
                }
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-z",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "{\"q\":"}
                    }]
                }
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-z",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": "\"hi\"}"}
                    }]
                }
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "chatcmpl-z",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }),
    ));

    // Exactly one tool_use start, with the id from the first chunk.
    let starts: Vec<_> = all
        .iter()
        .filter_map(|e| match e {
            AnthropicEvent::ContentBlockStart { content_block, .. } => Some(content_block),
            _ => None,
        })
        .collect();
    assert_eq!(starts.len(), 1);
    if let StreamingContentBlock::ToolUse { id, name, .. } = starts[0] {
        assert_eq!(id, "chatcmpl-tool-abc");
        assert_eq!(name, "search");
    } else {
        panic!("expected tool_use start");
    }

    // Fragments concatenate to the full JSON.
    let mut buf = String::new();
    for ev in &all {
        if let AnthropicEvent::ContentBlockDelta {
            delta: BlockDelta::InputJsonDelta { partial_json },
            ..
        } = ev
        {
            buf.push_str(partial_json);
        }
    }
    assert_eq!(buf, "{\"q\":\"hi\"}");
}

#[test]
fn vllm_chatcmpl_tool_prefix_id_passes_through() {
    // vLLM uses `chatcmpl-tool-<hex>` instead of `call_<hex>`. We pass it
    // through unchanged so downstream agents can correlate.
    let raw = json!({
        "id": "chatcmpl-x",
        "model": "Qwen",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "chatcmpl-tool-9f3c2a0b1d3e4f56",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let out = translate_response(raw);
    assert_eq!(out["content"][0]["id"], "chatcmpl-tool-9f3c2a0b1d3e4f56");
}

// ---------- SGLang ----------

#[test]
fn sglang_null_reasoning_content_in_every_delta_is_ignored() {
    // SGLang emits `reasoning_content: null` on every SSE chunk. Our
    // translator must NOT treat the field's presence-as-null as a signal
    // to open a thinking block.
    let mut t = translator_native();
    let mut all = Vec::new();
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": null, "role": "assistant", "content": "hi"},
                "finish_reason": null,
                "matched_stop": null
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": null, "content": " there"}
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }),
    ));
    // No thinking block opened.
    let opened_thinking = all.iter().any(|e| matches!(e,
        AnthropicEvent::ContentBlockStart {
            content_block: StreamingContentBlock::Thinking { .. }, ..
        }));
    assert!(!opened_thinking, "null reasoning_content should NOT open a thinking block");
}

#[test]
fn sglang_null_id_and_name_on_continuation_tool_call_chunks() {
    // SGLang sends `id: null` and `function.name: null` on continuation
    // chunks (not omitted, but explicitly null). Our deserializer treats
    // them as Option<String>=None, which is correct.
    let mut t = translator_native();
    let mut all = Vec::new();
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "id": "call_5a8b3e2f",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "search", "arguments": ""}
                    }]
                }
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "id": null,
                        "index": 0,
                        "type": "function",
                        "function": {"name": null, "arguments": "{\"q\":\"x\"}"}
                    }]
                }
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
        }),
    ));

    let starts: Vec<_> = all
        .iter()
        .filter(|e| matches!(e, AnthropicEvent::ContentBlockStart { .. }))
        .collect();
    assert_eq!(starts.len(), 1, "exactly one tool_use block opened");
    let mut buf = String::new();
    for ev in &all {
        if let AnthropicEvent::ContentBlockDelta {
            delta: BlockDelta::InputJsonDelta { partial_json },
            ..
        } = ev
        {
            buf.push_str(partial_json);
        }
    }
    assert_eq!(buf, "{\"q\":\"x\"}");
}

#[test]
fn sglang_matched_stop_field_and_top_level_metadata_are_ignored() {
    // SGLang adds top-level metadata + sglext + per-choice matched_stop.
    let raw = json!({
        "id": "abc",
        "model": "Qwen",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok", "reasoning_content": null},
            "finish_reason": "stop",
            "matched_stop": "<|im_end|>"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6, "reasoning_tokens": 0},
        "metadata": {"weight_version": "v42"},
        "sglext": {"cached_tokens_details": {"device": 0, "host": 0}}
    });
    let out = translate_response(raw);
    assert_eq!(out["content"][0]["text"], "ok");
}

#[test]
fn sglang_finish_reason_abort_maps_to_end_turn() {
    // SGLang adds the `"abort"` finish_reason. We treat unknowns as
    // end_turn (matching LiteLLM's permissive default).
    let raw = json!({
        "id": "abc",
        "model": "Qwen",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "partial"},
            "finish_reason": "abort"
        }]
    });
    let out = translate_response(raw);
    assert_eq!(out["stop_reason"], "end_turn");
}

#[test]
fn sglang_reasoning_content_stream_via_real_field_opens_thinking_block() {
    // When `reasoning_content` is a string (not null), we open a thinking
    // block. (Only the null-everywhere case from the previous test gets
    // ignored.)
    let mut t = translator_native();
    let mut all = Vec::new();
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "Let me think...", "role": "assistant"}
            }]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{"index": 0, "delta": {"reasoning_content": null, "content": "42"}}]
        }),
    ));
    all.extend(push(
        &mut t,
        json!({
            "id": "abc",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }),
    ));
    let opened_thinking = all.iter().any(|e| matches!(e,
        AnthropicEvent::ContentBlockStart {
            content_block: StreamingContentBlock::Thinking { .. }, ..
        }));
    assert!(opened_thinking);
}

#[test]
fn sglang_kimi_k2_tool_id_format_passes_through() {
    // SGLang's kimi_k2 parser uses `functions.<name>:<int>` IDs. We must
    // accept them on input and emit them back unchanged.
    let raw = json!({
        "id": "abc",
        "model": "kimi-k2",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "functions.search:0",
                    "index": 0,
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let out = translate_response(raw);
    assert_eq!(out["content"][0]["id"], "functions.search:0");
    assert_eq!(out["content"][0]["type"], "tool_use");
    let _ = AnthropicStopReason::ToolUse; // keep import live
}
