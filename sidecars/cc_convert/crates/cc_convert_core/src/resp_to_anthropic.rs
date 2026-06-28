//! OpenAI Chat Completions response → Anthropic Messages response.

use crate::anthropic::*;
use crate::error::ConvertError;
use crate::openai::*;
use crate::tool_names::ToolNameMap;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct ResponseConvertOptions {
    /// If `Some`, use this string as the Anthropic response's `model` field.
    /// If `None`, pass through the OpenAI response's model (LiteLLM behaviour).
    pub original_model: Option<String>,
    /// LiteLLM passes `id` through as-is (e.g. `chatcmpl-xxx`). If true,
    /// rewrite `chatcmpl-` prefix to `msg_`. Default false (= LiteLLM).
    pub rewrite_id: bool,
    /// LiteLLM allows `content: []` for empty assistant messages. If true,
    /// always emit at least one `{type:"text", text:""}` block (older
    /// Anthropic SDKs require this). Default false (= LiteLLM).
    pub never_empty_content: bool,
    /// LiteLLM subtracts `cached_tokens` from `prompt_tokens` so
    /// `input_tokens` only counts the uncached portion. Default true.
    pub subtract_cached_from_input: bool,
}

impl Default for ResponseConvertOptions {
    fn default() -> Self {
        Self {
            original_model: None,
            rewrite_id: false,
            never_empty_content: false,
            subtract_cached_from_input: true,
        }
    }
}

impl ResponseConvertOptions {
    /// Byte-for-byte parity with LiteLLM's
    /// `translate_openai_response_to_anthropic` (modulo dropped nulls).
    pub fn litellm_compat() -> Self {
        Self::default()
    }

    /// Friendlier for older Anthropic clients: rewrites `id` to `msg_*` and
    /// guarantees at least one content block.
    pub fn pragmatic(original_model: impl Into<String>) -> Self {
        Self {
            original_model: Some(original_model.into()),
            rewrite_id: true,
            never_empty_content: true,
            subtract_cached_from_input: true,
        }
    }
}

pub fn openai_response_to_anthropic(
    resp: &OpenAIResponse,
    original_model: &str,
    tool_name_map: &ToolNameMap,
) -> Result<AnthropicResponse, ConvertError> {
    // Backwards-compat wrapper: behaves like the old API (rewrites id +
    // guarantees non-empty content + uses `original_model`).
    let opts = ResponseConvertOptions::pragmatic(original_model);
    openai_response_to_anthropic_with(resp, tool_name_map, &opts)
}

pub fn openai_response_to_anthropic_with(
    resp: &OpenAIResponse,
    tool_name_map: &ToolNameMap,
    opts: &ResponseConvertOptions,
) -> Result<AnthropicResponse, ConvertError> {
    let id = if opts.rewrite_id {
        rewrite_id(&resp.id)
    } else {
        resp.id.clone()
    };

    let model = opts
        .original_model
        .clone()
        .unwrap_or_else(|| resp.model.clone());

    let choice = resp
        .choices
        .first()
        .ok_or_else(|| ConvertError::InvalidResponse("response has no choices".into()))?;

    let mut content = Vec::<ResponseContentBlock>::new();

    if let Some(reasoning) = choice
        .message
        .reasoning_content
        .as_ref()
        .filter(|s| !s.is_empty())
    {
        content.push(ResponseContentBlock::Thinking {
            thinking: reasoning.clone(),
            signature: None,
        });
    }

    if let Some(text) = choice.message.content.as_ref().filter(|s| !s.is_empty()) {
        content.push(ResponseContentBlock::Text { text: text.clone() });
    }

    if let Some(tool_calls) = choice.message.tool_calls.as_ref() {
        for tc in tool_calls {
            let restored = tool_name_map.restore(tc.function.name.as_deref().unwrap_or(""));
            let input: Value = match tc.function.arguments.as_deref().unwrap_or("") {
                "" => json!({}),
                raw => serde_json::from_str(raw).unwrap_or_else(|_| json!({ "raw": raw })),
            };
            content.push(ResponseContentBlock::ToolUse {
                id: tc.id.clone().unwrap_or_default(),
                name: restored.to_string(),
                input,
            });
        }
    }

    if content.is_empty() && opts.never_empty_content {
        content.push(ResponseContentBlock::Text { text: String::new() });
    }

    let stop_reason = map_stop_reason(choice.finish_reason.as_deref());
    let usage = map_usage(resp.usage.as_ref(), opts.subtract_cached_from_input);

    Ok(AnthropicResponse {
        id,
        msg_type: "message".to_string(),
        role: "assistant".to_string(),
        model,
        content,
        stop_reason,
        stop_sequence: None,
        usage,
    })
}

pub fn rewrite_id(openai_id: &str) -> String {
    if let Some(rest) = openai_id.strip_prefix("chatcmpl-") {
        format!("msg_{}", rest)
    } else if openai_id.starts_with("msg_") {
        openai_id.to_string()
    } else {
        format!("msg_{}", openai_id)
    }
}

pub fn map_stop_reason(reason: Option<&str>) -> AnthropicStopReason {
    match reason {
        Some("stop") => AnthropicStopReason::EndTurn,
        Some("length") => AnthropicStopReason::MaxTokens,
        Some("tool_calls") => AnthropicStopReason::ToolUse,
        Some("function_call") => AnthropicStopReason::ToolUse, // legacy
        _ => AnthropicStopReason::EndTurn,
    }
}

pub fn map_usage(usage: Option<&OpenAIUsage>, subtract_cached: bool) -> AnthropicUsage {
    let Some(u) = usage else {
        return AnthropicUsage::default();
    };
    let cache_read = u
        .prompt_tokens_details
        .as_ref()
        .and_then(|d| d.cached_tokens);
    let mut input_tokens = u.prompt_tokens;
    if subtract_cached {
        if let Some(c) = cache_read {
            input_tokens = input_tokens.saturating_sub(c);
        }
    }
    AnthropicUsage {
        input_tokens,
        output_tokens: u.completion_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cache_read,
    }
}
