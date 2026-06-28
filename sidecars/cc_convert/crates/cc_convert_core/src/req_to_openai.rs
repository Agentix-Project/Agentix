//! Anthropic Messages request → OpenAI Chat Completions request.

use crate::anthropic::*;
use crate::error::ConvertError;
use crate::openai::*;
use crate::tool_names::ToolNameMap;
use serde_json::{json, Value};

/// How to forward Anthropic `thinking` blocks (assistant-side reasoning text)
/// when translating prior turns to OpenAI Chat Completions input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningPassthrough {
    /// Omit any reasoning info on the assistant message.
    /// Use this for the DeepSeek hosted API, which returns HTTP 400 when
    /// `reasoning_content` is present on input.
    Drop,
    /// Collapse all `thinking` block texts into a single `reasoning_content`
    /// string on the assistant message. This is the real wire-format
    /// understood by vLLM (`reasoning` alias), SGLang, and consumed by the
    /// Qwen3 chat template. All other upstreams silently ignore the unknown
    /// field. This is the default.
    ReasoningContent,
    /// Preserve LiteLLM's intermediate `thinking_blocks` array. LiteLLM
    /// itself never wire-sends this (its provider transformations strip it),
    /// so only use this when you need to be a literal drop-in for LiteLLM's
    /// `AnthropicAdapter` intermediate output.
    LiteLLMThinkingBlocks,
}

#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Drop Anthropic-only `cache_control` fields (always true for OpenAI targets).
    pub drop_cache_control: bool,
    /// Drop Anthropic-only `top_k` field.
    pub drop_top_k: bool,
    /// When `stream: true`, inject `stream_options: {include_usage: true}`.
    /// Default off so byte-level parity with LiteLLM holds; turn on for real-world use.
    pub inject_include_usage: bool,
    /// Rewrite `max_tokens` → `max_completion_tokens` for reasoning models
    /// (o1/o3/o4/gpt-5). LiteLLM does NOT do this; default off for parity.
    pub use_max_completion_tokens_for_reasoning_models: bool,
    /// Collapse a single-text system/user content block into a plain string.
    /// LiteLLM keeps it as a list — default off for parity.
    pub collapse_single_text_part: bool,
    /// Concatenate MULTI-text content blocks into one string with "\n\n"
    /// between parts (then emit as a plain string). Many real OpenAI-compat
    /// servers (SGLang/vLLM in strict mode) reject list-content on system
    /// and pure-text user messages. Off in litellm_compat, on in pragmatic.
    pub concat_multi_text_parts: bool,
    /// Emit `stop` (OpenAI-spec). LiteLLM forwards `stop_sequences` verbatim
    /// because its downstream call layer does the rename — default false
    /// (passthrough) for parity.
    pub emit_stop_field: bool,
    /// How to forward Anthropic `thinking` blocks on prior assistant
    /// messages. See [`ReasoningPassthrough`]. Default
    /// `ReasoningContent` — the only spelling consumed by real upstreams.
    pub reasoning_passthrough: ReasoningPassthrough,
    /// Drop messages whose only content is an empty string (LiteLLM behaviour).
    pub drop_empty_string_messages: bool,
    /// Override the target model name. None → use the request's model verbatim.
    pub target_model: Option<String>,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            drop_cache_control: true,
            drop_top_k: false,
            inject_include_usage: false,
            use_max_completion_tokens_for_reasoning_models: false,
            collapse_single_text_part: false,
            concat_multi_text_parts: false,
            emit_stop_field: false,
            reasoning_passthrough: ReasoningPassthrough::ReasoningContent,
            drop_empty_string_messages: true,
            target_model: None,
        }
    }
}

impl ConvertOptions {
    /// Preset for byte-equivalent parity with LiteLLM's
    /// `AnthropicAdapter.translate_anthropic_to_openai`. Emits LiteLLM's
    /// intermediate `thinking_blocks` shape on assistant messages (LiteLLM
    /// itself strips this in its provider transformations before send;
    /// real upstreams silently ignore the unknown field).
    pub fn litellm_compat() -> Self {
        Self {
            reasoning_passthrough: ReasoningPassthrough::LiteLLMThinkingBlocks,
            ..Self::default()
        }
    }

    /// Preset for real OAI-compat upstreams (SGLang/vLLM strict mode etc.):
    /// collapse single-text content to a string AND concat multi-text-block
    /// content with "\n\n" so the wire format is always a string when no
    /// multimodal parts are present. Drops top_k (real OpenAI rejects it),
    /// rewrites `stop_sequences` → `stop`, swaps in `max_completion_tokens`
    /// for reasoning models, injects `stream_options.include_usage`. Forwards
    /// reasoning as `reasoning_content` so prior thinking flows to Qwen3 /
    /// vLLM / SGLang chat templates that actually consume it.
    pub fn pragmatic() -> Self {
        Self {
            drop_cache_control: true,
            drop_top_k: true,
            inject_include_usage: true,
            use_max_completion_tokens_for_reasoning_models: true,
            collapse_single_text_part: true,
            concat_multi_text_parts: true,
            emit_stop_field: true,
            reasoning_passthrough: ReasoningPassthrough::ReasoningContent,
            drop_empty_string_messages: true,
            target_model: None,
        }
    }
}

/// Returns the OpenAI request and a tool-name map (which the response
/// translator needs to restore the original Anthropic names).
pub fn anthropic_request_to_openai(
    req: &AnthropicRequest,
    opts: &ConvertOptions,
) -> Result<(OpenAIRequest, ToolNameMap), ConvertError> {
    let model = opts.target_model.clone().unwrap_or_else(|| req.model.clone());
    let uses_max_completion_tokens =
        opts.use_max_completion_tokens_for_reasoning_models && is_reasoning_model(&model);

    let mut messages: Vec<OpenAIMessage> = Vec::new();

    // 1) system → leading system message
    if let Some(sys) = &req.system {
        match sys {
            SystemField::Text(s) => {
                if !s.is_empty() {
                    messages.push(OpenAIMessage::System {
                        content: OpenAIContent::Text(s.clone()),
                    });
                }
            }
            SystemField::Blocks(blocks) => {
                let parts: Vec<OpenAIContentPart> = blocks
                    .iter()
                    .map(|b| OpenAIContentPart::Text { text: b.text.clone() })
                    .collect();
                if !parts.is_empty() {
                    let content = if opts.collapse_single_text_part && parts.len() == 1 {
                        if let OpenAIContentPart::Text { text } = &parts[0] {
                            OpenAIContent::Text(text.clone())
                        } else {
                            OpenAIContent::Parts(parts)
                        }
                    } else if opts.concat_multi_text_parts
                        && parts.iter().all(|p| matches!(p, OpenAIContentPart::Text { .. }))
                    {
                        // All text — concat with blank line between.
                        let joined = parts
                            .iter()
                            .map(|p| match p {
                                OpenAIContentPart::Text { text } => text.as_str(),
                                _ => "",
                            })
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        OpenAIContent::Text(joined)
                    } else {
                        OpenAIContent::Parts(parts)
                    };
                    messages.push(OpenAIMessage::System { content });
                }
            }
        }
    }

    // 2) messages → user/assistant/tool messages
    for msg in &req.messages {
        translate_message(msg, &mut messages, opts)?;
    }

    // 3) tools + name map. Hosted Anthropic tools (web_search_*, computer_*,
    // bash_*, text_editor_*, web_fetch_*, code_execution_*, tool_search_*)
    // have no OpenAI equivalent — drop them rather than forwarding shapes
    // OpenAI will reject with HTTP 400. Client tools translate normally.
    let mut tool_name_map = ToolNameMap::new();
    let tools = req.tools.as_ref().and_then(|tools| {
        let translated: Vec<OpenAITool> = tools
            .iter()
            .filter_map(|t| match t {
                AnthropicTool::Client(c) => Some(OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunctionDef {
                        name: tool_name_map.translate(&c.name),
                        description: c.description.clone(),
                        parameters: c.input_schema.clone(),
                    },
                }),
                AnthropicTool::Hosted(_h) => {
                    // Drop. (No-op log point; future: collect into a
                    // translation_warnings sidechannel for /v1/messages 200s.)
                    None
                }
            })
            .collect();
        if translated.is_empty() {
            None
        } else {
            Some(translated)
        }
    });

    // 4) tool_choice translation
    let tool_choice = req.tool_choice.as_ref().map(|tc| match tc {
        AnthropicToolChoice::Auto { .. } => json!("auto"),
        AnthropicToolChoice::Any { .. } => json!("required"),
        AnthropicToolChoice::None => json!("none"),
        AnthropicToolChoice::Tool { name, .. } => {
            let translated = tool_name_map
                .0
                .iter()
                .find(|(_, v)| v.as_str() == name.as_str())
                .map(|(k, _)| k.clone())
                .unwrap_or_else(|| name.clone());
            json!({"type": "function", "function": {"name": translated}})
        }
    });

    // 5) thinking → reasoning_effort.
    // Two possible sources, in priority order:
    //   (a) output_config.effort (Anthropic 2025 Q4 extension used by Claude
    //       Code / OpenCode / Cline — an explicit "low"/"medium"/"high"
    //       string the client picked itself). Wins when present.
    //   (b) thinking.budget_tokens (older field) — bucketed.
    let explicit_effort = req
        .extra
        .get("output_config")
        .and_then(|v| v.get("effort"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let reasoning_effort = explicit_effort.or_else(|| {
        req.thinking.as_ref().and_then(|t| match t {
            AnthropicThinking::Enabled { budget_tokens } => Some(bucket_reasoning_effort(*budget_tokens)),
            AnthropicThinking::Disabled => None,
        })
    });

    // 6) stream_options injection
    let stream_options = match (req.stream, opts.inject_include_usage) {
        (Some(true), true) => Some(StreamOptions { include_usage: true }),
        _ => None,
    };

    // 7) service_tier: Anthropic "auto"/"standard_only" → OpenAI tier names.
    let service_tier = req
        .extra
        .get("service_tier")
        .and_then(|v| v.as_str())
        .map(|s| match s {
            // Anthropic spec values
            "auto" => "auto".to_string(),
            "standard_only" => "default".to_string(),
            // OpenAI native values — pass through verbatim
            other => other.to_string(),
        });

    let openai_req = OpenAIRequest {
        model,
        messages,
        max_tokens: if uses_max_completion_tokens { None } else { Some(req.max_tokens) },
        max_completion_tokens: if uses_max_completion_tokens { Some(req.max_tokens) } else { None },
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: if opts.drop_top_k { None } else { req.top_k },
        stop: if opts.emit_stop_field { req.stop_sequences.clone() } else { None },
        stop_sequences: if opts.emit_stop_field { None } else { req.stop_sequences.clone() },
        stream: req.stream,
        stream_options,
        tools,
        tool_choice,
        user: req.metadata.as_ref().and_then(|m| m.user_id.clone()),
        reasoning_effort,
        service_tier,
    };

    let _ = opts.drop_cache_control;

    Ok((openai_req, tool_name_map))
}

fn translate_message(
    msg: &AnthropicMessage,
    out: &mut Vec<OpenAIMessage>,
    opts: &ConvertOptions,
) -> Result<(), ConvertError> {
    // Bare-string content keeps its string shape on the OpenAI side
    // (matches LiteLLM and is what most providers expect).
    let bare_text: Option<String> = match &msg.content {
        MessageContent::Text(s) => Some(s.clone()),
        MessageContent::Blocks(_) => None,
    };

    let blocks: Vec<ContentBlock> = match &msg.content {
        MessageContent::Text(s) => vec![ContentBlock::Text {
            text: s.clone(),
            cache_control: None,
        }],
        MessageContent::Blocks(b) => b.clone(),
    };

    match msg.role.as_str() {
        "user" => translate_user_blocks(&blocks, bare_text, out, opts)?,
        "assistant" => translate_assistant_blocks(&blocks, bare_text, out, opts)?,
        other => {
            return Err(ConvertError::InvalidRequest(format!(
                "unknown message role: {other}"
            )))
        }
    }
    Ok(())
}

fn translate_user_blocks(
    blocks: &[ContentBlock],
    bare_text: Option<String>,
    out: &mut Vec<OpenAIMessage>,
    opts: &ConvertOptions,
) -> Result<(), ConvertError> {
    // tool_result blocks must each become their own role="tool" message,
    // emitted BEFORE any trailing user content (matches LiteLLM ordering).
    let mut user_parts: Vec<OpenAIContentPart> = Vec::new();

    for b in blocks {
        match b {
            ContentBlock::Text { text, .. } => {
                user_parts.push(OpenAIContentPart::Text { text: text.clone() })
            }
            ContentBlock::Image { source, .. } => {
                user_parts.push(OpenAIContentPart::ImageUrl {
                    image_url: OpenAIImageUrl {
                        url: image_source_to_url(source),
                    },
                });
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                let content = tool_result_to_openai_content(content.as_ref());
                out.push(OpenAIMessage::Tool {
                    tool_call_id: tool_use_id.clone(),
                    content,
                });
            }
            ContentBlock::ToolUse { .. } => {
                return Err(ConvertError::InvalidRequest(
                    "tool_use block found in user message".into(),
                ))
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {
                // Thinking blocks are assistant-only; ignore in user.
            }
            ContentBlock::Unknown => {
                // Server-side content blocks (web_search_tool_result,
                // code_execution_tool_result, mcp_tool_use, mcp_tool_result,
                // server_tool_use, container_upload, document, etc.) have
                // no OpenAI equivalent. Silently drop so the upstream
                // doesn't 400 on the unknown content shape.
            }
        }
    }

    if let Some(text) = bare_text {
        if text.is_empty() && opts.drop_empty_string_messages {
            return Ok(());
        }
        // Pure bare-string user message → string content (LiteLLM shape).
        out.push(OpenAIMessage::User {
            content: OpenAIContent::Text(text),
        });
        return Ok(());
    }

    if !user_parts.is_empty() {
        let content = collapse_parts(user_parts, opts);
        out.push(OpenAIMessage::User { content });
    }
    Ok(())
}

fn translate_assistant_blocks(
    blocks: &[ContentBlock],
    bare_text: Option<String>,
    out: &mut Vec<OpenAIMessage>,
    opts: &ConvertOptions,
) -> Result<(), ConvertError> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
    // Two accumulators — we pick which one to emit based on
    // opts.reasoning_passthrough at the end.
    let mut reasoning_texts: Vec<String> = Vec::new();
    let mut thinking_blocks_raw: Vec<Value> = Vec::new();

    for b in blocks {
        match b {
            ContentBlock::Text { text, .. } => text_parts.push(text.clone()),
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(OpenAIToolCall {
                    id: Some(id.clone()),
                    call_type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: Some(name.clone()),
                        arguments: Some(serde_json::to_string(input)?),
                    },
                    index: None,
                });
            }
            ContentBlock::Thinking { thinking, signature } => {
                if opts.reasoning_passthrough != ReasoningPassthrough::Drop {
                    reasoning_texts.push(thinking.clone());
                    if opts.reasoning_passthrough
                        == ReasoningPassthrough::LiteLLMThinkingBlocks
                    {
                        let mut o = serde_json::Map::new();
                        o.insert("type".to_string(), json!("thinking"));
                        o.insert("thinking".to_string(), json!(thinking));
                        if let Some(sig) = signature {
                            o.insert("signature".to_string(), json!(sig));
                        }
                        // LiteLLM AnthropicAdapter adds an empty cache_control
                        // here; mirror exactly for byte-parity.
                        o.insert("cache_control".to_string(), json!({}));
                        thinking_blocks_raw.push(Value::Object(o));
                    }
                }
            }
            ContentBlock::RedactedThinking { data } => {
                if opts.reasoning_passthrough
                    == ReasoningPassthrough::LiteLLMThinkingBlocks
                {
                    thinking_blocks_raw.push(json!({
                        "type": "redacted_thinking",
                        "data": data,
                    }));
                }
                // ReasoningPassthrough::ReasoningContent: no plain-text
                // representation for redacted blocks, drop.
            }
            ContentBlock::Image { .. } | ContentBlock::ToolResult { .. } => {
                return Err(ConvertError::InvalidRequest(
                    "image/tool_result block in assistant message".into(),
                ))
            }
            ContentBlock::Unknown => {
                // server_tool_use, mcp_tool_use, code_execution_tool_result,
                // bash_code_execution_tool_result, etc. — Anthropic
                // server-side blocks with no OpenAI equivalent. Drop.
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(OpenAIContent::Text(text_parts.join("")))
    };
    let tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let (reasoning_content, thinking_blocks) = match opts.reasoning_passthrough {
        ReasoningPassthrough::Drop => (None, None),
        ReasoningPassthrough::ReasoningContent => {
            let rc = if reasoning_texts.is_empty() {
                None
            } else {
                Some(reasoning_texts.join("\n\n"))
            };
            (rc, None)
        }
        ReasoningPassthrough::LiteLLMThinkingBlocks => {
            let tb = if thinking_blocks_raw.is_empty() {
                None
            } else {
                Some(thinking_blocks_raw)
            };
            (None, tb)
        }
    };

    if content.is_some()
        || tool_calls.is_some()
        || reasoning_content.is_some()
        || thinking_blocks.is_some()
    {
        out.push(OpenAIMessage::Assistant {
            content,
            tool_calls,
            reasoning_content,
            thinking_blocks,
        });
    }
    let _ = bare_text;
    Ok(())
}

fn collapse_parts(parts: Vec<OpenAIContentPart>, opts: &ConvertOptions) -> OpenAIContent {
    if opts.collapse_single_text_part && parts.len() == 1 {
        if let OpenAIContentPart::Text { text } = &parts[0] {
            return OpenAIContent::Text(text.clone());
        }
    }
    if opts.concat_multi_text_parts
        && parts.iter().all(|p| matches!(p, OpenAIContentPart::Text { .. }))
    {
        let joined = parts
            .iter()
            .map(|p| match p {
                OpenAIContentPart::Text { text } => text.as_str(),
                _ => "",
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        return OpenAIContent::Text(joined);
    }
    OpenAIContent::Parts(parts)
}

fn image_source_to_url(src: &ImageSource) -> String {
    match src {
        ImageSource::Base64 { media_type, data } => {
            format!("data:{};base64,{}", media_type, data)
        }
        ImageSource::Url { url } => url.clone(),
    }
}

fn tool_result_to_openai_content(content: Option<&ToolResultContent>) -> OpenAIContent {
    match content {
        None => OpenAIContent::Text(String::new()),
        Some(ToolResultContent::Text(s)) => OpenAIContent::Text(s.clone()),
        Some(ToolResultContent::Blocks(blocks)) => {
            // Single text block → flatten. Anything else (image, multi-block) →
            // list-content under the same tool_call_id (matches LiteLLM and the
            // Anthropic 1:1 tool_use_id ↔ tool message rule).
            if blocks.len() == 1 {
                if let ToolResultBlock::Text { text } = &blocks[0] {
                    return OpenAIContent::Text(text.clone());
                }
            }
            let parts: Vec<OpenAIContentPart> = blocks
                .iter()
                .map(|b| match b {
                    ToolResultBlock::Text { text } => OpenAIContentPart::Text { text: text.clone() },
                    ToolResultBlock::Image { source } => OpenAIContentPart::ImageUrl {
                        image_url: OpenAIImageUrl {
                            url: image_source_to_url(source),
                        },
                    },
                })
                .collect();
            OpenAIContent::Parts(parts)
        }
    }
}

fn is_reasoning_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // OpenAI reasoning families: o1/o3/o4 series, gpt-5 series.
    m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4") || m.starts_with("gpt-5")
}

fn bucket_reasoning_effort(budget_tokens: u32) -> String {
    if budget_tokens >= 10_000 {
        "high".to_string()
    } else if budget_tokens >= 5_000 {
        "medium".to_string()
    } else if budget_tokens >= 2_000 {
        "low".to_string()
    } else {
        "minimal".to_string()
    }
}

/// Helper used by the streaming layer and the JSON facade.
pub fn json_value_of_openai_request(req: &OpenAIRequest) -> Value {
    serde_json::to_value(req).expect("OpenAIRequest serialises")
}
