//! Cases 1–20 from the plan: request translation (Anthropic → OpenAI).

use cc_convert_core::anthropic::AnthropicRequest;
use cc_convert_core::{anthropic_request_to_openai, ConvertOptions};
use serde_json::{json, Value};

/// Translate via the **pragmatic** options (single-text collapse, max_completion_tokens
/// for reasoning models, stream_options.include_usage). The LiteLLM-parity test suite
/// uses `ConvertOptions::litellm_compat()` instead.
fn convert(req_json: Value) -> (Value, Value) {
    let req: AnthropicRequest =
        serde_json::from_value(req_json).expect("parse anthropic request");
    let (openai_req, tool_map) =
        anthropic_request_to_openai(&req, &ConvertOptions::pragmatic()).expect("translate");
    let openai_value = serde_json::to_value(&openai_req).expect("serialize openai");
    let map_value = serde_json::to_value(&tool_map).expect("serialize tool map");
    (openai_value, map_value)
}

#[test]
fn case01_plain_user_text() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}]
    }));
    assert_eq!(out["model"], "gpt-4o-mini");
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["messages"][0]["role"], "user");
    assert_eq!(out["messages"][0]["content"], "hello");
}

#[test]
fn case02_system_string() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "system": "Be concise.",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(out["messages"][0]["role"], "system");
    assert_eq!(out["messages"][0]["content"], "Be concise.");
    assert_eq!(out["messages"][1]["role"], "user");
}

#[test]
fn case03_system_blocks_with_cache_control_drops_cache_control() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "system": [
            {"type": "text", "text": "rule 1", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(out["messages"][0]["role"], "system");
    // Single-text block collapses to a string.
    assert_eq!(out["messages"][0]["content"], "rule 1");
    // No cache_control survives on the OpenAI side.
    assert!(serde_json::to_string(&out).unwrap().find("cache_control").is_none());
}

#[test]
fn case04_multi_turn() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "ok"}
        ]
    }));
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["content"], "hello!");
}

#[test]
fn case05_user_image_base64() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                {"type": "text", "text": "what is this?"}
            ]
        }]
    }));
    let parts = out["messages"][0]["content"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "image_url");
    assert_eq!(parts[0]["image_url"]["url"], "data:image/png;base64,AAAA");
    assert_eq!(parts[1]["type"], "text");
}

#[test]
fn case06_user_image_url() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
            ]
        }]
    }));
    let parts = out["messages"][0]["content"].as_array().unwrap();
    assert_eq!(parts[0]["image_url"]["url"], "https://example.com/x.png");
}

#[test]
fn case07_assistant_single_tool_use() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
            ]
        }]
    }));
    let tc = &out["messages"][0]["tool_calls"][0];
    assert_eq!(tc["id"], "toolu_1");
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(tc["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Paris"}));
}

#[test]
fn case08_assistant_two_parallel_tool_uses() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu_a", "name": "f1", "input": {"a": 1}},
                {"type": "tool_use", "id": "tu_b", "name": "f2", "input": {"b": 2}}
            ]
        }]
    }));
    let tcs = out["messages"][0]["tool_calls"].as_array().unwrap();
    assert_eq!(tcs.len(), 2);
    assert_eq!(tcs[0]["id"], "tu_a");
    assert_eq!(tcs[1]["id"], "tu_b");
}

#[test]
fn case09_user_single_tool_result() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "21C"}
            ]
        }]
    }));
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "tool");
    assert_eq!(msgs[0]["tool_call_id"], "toolu_1");
    assert_eq!(msgs[0]["content"], "21C");
}

#[test]
fn case10_user_three_tool_results_emit_three_messages() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "b"},
                {"type": "tool_result", "tool_use_id": "t3", "content": "c"}
            ]
        }]
    }));
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 3);
    for (i, expected_id) in ["t1", "t2", "t3"].iter().enumerate() {
        assert_eq!(msgs[i]["role"], "tool");
        assert_eq!(msgs[i]["tool_call_id"], *expected_id);
    }
}

#[test]
fn case11_user_tool_result_multipart_keeps_one_message() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "text", "text": "see image:"},
                    {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
                ]}
            ]
        }]
    }));
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0]["role"], "tool");
    let parts = msgs[0]["content"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[1]["type"], "image_url");
}

#[test]
fn case12_tools_input_schema_passthrough() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "name": "get_weather",
            "description": "weather lookup",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }]
    }));
    let tool = &out["tools"][0];
    assert_eq!(tool["type"], "function");
    assert_eq!(tool["function"]["name"], "get_weather");
    assert_eq!(tool["function"]["description"], "weather lookup");
    assert_eq!(tool["function"]["parameters"]["properties"]["city"]["type"], "string");
}

#[test]
fn case13_long_tool_name_truncated() {
    let long_name: String = "x".repeat(80);
    let (out, map) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "name": long_name,
            "input_schema": {"type": "object"}
        }]
    }));
    let translated = out["tools"][0]["function"]["name"].as_str().unwrap();
    assert!(translated.len() <= 64);
    assert!(translated.starts_with("x"));
    let map_obj = map.as_object().unwrap();
    assert_eq!(map_obj.get(translated).unwrap().as_str().unwrap(), &long_name);
}

#[test]
fn case14_tool_choice_any_becomes_required() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": {"type": "any"}
    }));
    assert_eq!(out["tool_choice"], "required");
}

#[test]
fn case15_tool_choice_named() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "f", "input_schema": {"type":"object"}}],
        "tool_choice": {"type": "tool", "name": "f"}
    }));
    assert_eq!(out["tool_choice"]["type"], "function");
    assert_eq!(out["tool_choice"]["function"]["name"], "f");
}

#[test]
fn case16_metadata_user_id() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": "u-123"}
    }));
    assert_eq!(out["user"], "u-123");
}

#[test]
fn case17_thinking_budget_bucketed_to_medium() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 5000}
    }));
    assert_eq!(out["reasoning_effort"], "medium");
}

#[test]
fn case18_top_k_dropped() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 20
    }));
    assert!(out.get("top_k").is_none());
}

#[test]
fn case19_stream_injects_include_usage() {
    let (out, _) = convert(json!({
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    }));
    assert_eq!(out["stream"], true);
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn case20_o_series_uses_max_completion_tokens() {
    let (out, _) = convert(json!({
        "model": "o3-mini",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "hi"}]
    }));
    assert_eq!(out["max_completion_tokens"], 200);
    assert!(out.get("max_tokens").is_none());
}

#[test]
fn thinking_history_becomes_reasoning_content_in_pragmatic() {
    // Anthropic prior-turn assistant message with a `thinking` block →
    // pragmatic mode should emit `reasoning_content: <text>` on the
    // assistant message and NOT emit `thinking_blocks` (LiteLLM-internal
    // shape that no real upstream consumes).
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hard math problem"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Step 1...\nStep 2...", "signature": "sig_x"},
                    {"type": "text", "text": "The answer is 42."}
                ]
            },
            {"role": "user", "content": "Why?"}
        ]
    }));
    let assistant = &out["messages"][1];
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["content"], "The answer is 42.");
    assert_eq!(assistant["reasoning_content"], "Step 1...\nStep 2...");
    assert!(
        assistant.get("thinking_blocks").is_none(),
        "thinking_blocks must not appear in pragmatic mode"
    );
}

#[test]
fn thinking_history_becomes_thinking_blocks_in_litellm_compat() {
    use cc_convert_core::ReasoningPassthrough;
    let mut opts = ConvertOptions::litellm_compat();
    opts.reasoning_passthrough = ReasoningPassthrough::LiteLLMThinkingBlocks;
    let req: AnthropicRequest = serde_json::from_value(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "thoughts", "signature": "sig_x"},
                    {"type": "text", "text": "A."}
                ]
            }
        ]
    }))
    .unwrap();
    let (req2, _) = anthropic_request_to_openai(&req, &opts).unwrap();
    let v = serde_json::to_value(&req2).unwrap();
    let asst = &v["messages"][1];
    assert!(asst.get("reasoning_content").is_none());
    let tb = asst["thinking_blocks"].as_array().unwrap();
    assert_eq!(tb[0]["type"], "thinking");
    assert_eq!(tb[0]["thinking"], "thoughts");
    assert_eq!(tb[0]["signature"], "sig_x");
}

#[test]
fn thinking_drop_mode_omits_both_fields() {
    use cc_convert_core::ReasoningPassthrough;
    let mut opts = ConvertOptions::pragmatic();
    opts.reasoning_passthrough = ReasoningPassthrough::Drop;
    let req: AnthropicRequest = serde_json::from_value(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "secret", "signature": "s"},
                {"type": "text", "text": "visible"}
            ]}
        ]
    }))
    .unwrap();
    let (req2, _) = anthropic_request_to_openai(&req, &opts).unwrap();
    let v = serde_json::to_value(&req2).unwrap();
    let asst = &v["messages"][0];
    assert_eq!(asst["content"], "visible");
    assert!(asst.get("reasoning_content").is_none());
    assert!(asst.get("thinking_blocks").is_none());
}

#[test]
fn hosted_tools_are_dropped_not_forwarded() {
    // Anthropic hosted tools (web_search_*, computer_*, bash_*, etc.) have
    // NO OpenAI equivalent — forwarding them produces HTTP 400 because
    // OpenAI's tools array only accepts {type:"function"}. We drop them.
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "search the web"}],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5
            },
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": 1280,
                "display_height_px": 800
            },
            {
                "type": "bash_20250124",
                "name": "bash"
            },
            {
                "type": "text_editor_20250124",
                "name": "str_replace_editor"
            },
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        ]
    }));
    // Only the client tool survives; hosted ones are dropped.
    let tools = out["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 1, "only the client tool should remain");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "get_weather");
    // Verify the serialized request contains nothing from the hosted shapes
    let raw = serde_json::to_string(&out).unwrap();
    assert!(!raw.contains("web_search_20250305"));
    assert!(!raw.contains("computer_20241022"));
    assert!(!raw.contains("bash_20250124"));
    assert!(!raw.contains("text_editor_20250124"));
}

#[test]
fn all_hosted_tools_produces_no_tools_field() {
    // If EVERY tool is hosted, we should omit the tools field entirely
    // rather than send an empty array (which OpenAI rejects as a no-op).
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "web_search_20250305", "name": "web_search"}
        ]
    }));
    assert!(
        out.get("tools").is_none(),
        "tools field should be omitted entirely when all tools were hosted"
    );
}

#[test]
fn unknown_content_block_types_are_dropped_not_rejected() {
    // server_tool_use, web_search_tool_result, code_execution_tool_result,
    // mcp_tool_use, etc. — Anthropic-specific server-side content blocks
    // that have no OpenAI equivalent. Translator must drop them silently
    // rather than 400-ing the upstream or panicking on deserialization.
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search the web."},
                    {
                        "type": "server_tool_use",
                        "id": "stu_1",
                        "name": "web_search",
                        "input": {"query": "weather Tokyo"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "stu_1",
                        "content": [
                            {"type": "web_search_result", "url": "https://x/", "title": "T"}
                        ]
                    },
                    {"type": "text", "text": "Summarize."}
                ]
            }
        ]
    }));
    // Assistant message kept its text (no server_tool_use leaked to wire).
    assert_eq!(out["messages"][0]["role"], "assistant");
    assert_eq!(out["messages"][0]["content"], "Let me search the web.");
    // User message kept its text (no web_search_tool_result leaked).
    assert_eq!(out["messages"][1]["role"], "user");
    assert_eq!(out["messages"][1]["content"], "Summarize.");
    let raw = serde_json::to_string(&out).unwrap();
    assert!(!raw.contains("server_tool_use"));
    assert!(!raw.contains("web_search_tool_result"));
}

#[test]
fn unknown_top_level_fields_are_captured_in_extra_not_silently_dropped() {
    // Real Anthropic API clients (Claude Code, OpenCode, Cline, Anthropic
    // SDK) routinely send top-level fields beyond the documented schema:
    // output_config, context_management, speed, container, mcp_servers,
    // service_tier, inference_geo, diagnostics, betas, top-level
    // cache_control, etc. Before this fix they were silently dropped at
    // deserialization. Now they survive into AnthropicRequest.extra.
    let req: AnthropicRequest = serde_json::from_value(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "speed": "fast",
        "output_config": {"effort": "high", "task_budget": 5000},
        "context_management": {"edits": [{"type": "clear_tool_uses_20250919", "keep": 5}]},
        "container": {"id": "cnt_123", "skills": ["python"]},
        "inference_geo": "us-east-1",
        "service_tier": "standard_only",
        "mcp_servers": [{"name": "fs", "url": "http://x"}],
        "diagnostics": {"previous_message_id": "msg_xyz"},
        "betas": ["interleaved-thinking-2025-05-14"]
    }))
    .expect("unknown fields must NOT cause deserialization to fail");
    assert!(req.extra.contains_key("speed"));
    assert!(req.extra.contains_key("output_config"));
    assert!(req.extra.contains_key("context_management"));
    assert!(req.extra.contains_key("container"));
    assert!(req.extra.contains_key("inference_geo"));
    assert!(req.extra.contains_key("service_tier"));
    assert!(req.extra.contains_key("mcp_servers"));
    assert!(req.extra.contains_key("diagnostics"));
    assert!(req.extra.contains_key("betas"));
}

#[test]
fn output_config_effort_overrides_thinking_budget_bucket() {
    // Claude Code / OpenCode / Cline use `output_config.effort` to set
    // reasoning_effort directly; cc_convert should respect it INSTEAD of
    // the bucket derived from thinking.budget_tokens.
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        // Bucket would say "low" (3000 → low)
        "thinking": {"type": "enabled", "budget_tokens": 3000},
        // But the explicit effort says "high"
        "output_config": {"effort": "high"}
    }));
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn output_config_effort_works_without_thinking() {
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": "minimal"}
    }));
    assert_eq!(out["reasoning_effort"], "minimal");
}

#[test]
fn thinking_budget_still_works_when_no_explicit_effort() {
    // Backwards-compat: thinking.budget_tokens still buckets as before.
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 10000}
    }));
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn service_tier_anthropic_to_openai_mapping() {
    // Anthropic "standard_only" → OpenAI "default"
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "service_tier": "standard_only"
    }));
    assert_eq!(out["service_tier"], "default");

    // "auto" passes through
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "service_tier": "auto"
    }));
    assert_eq!(out["service_tier"], "auto");

    // OpenAI-native values (priority/flex/scale) pass through verbatim
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "service_tier": "priority"
    }));
    assert_eq!(out["service_tier"], "priority");
}

#[test]
fn metadata_user_id_passthrough_even_when_stringified_json() {
    // Claude Code stuffs {device_id, account_uuid, session_id} into the
    // user_id STRING as serialized JSON. We just pass it through to
    // OpenAI `user` verbatim — no parsing, no rejection.
    let claude_code_user_id = r#"{"device_id":"a3f7","account_uuid":"01HX","session_id":"d4e2"}"#;
    let (out, _) = convert(json!({
        "model": "claude-opus-4-7",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "metadata": {"user_id": claude_code_user_id}
    }));
    assert_eq!(out["user"], claude_code_user_id);
}
