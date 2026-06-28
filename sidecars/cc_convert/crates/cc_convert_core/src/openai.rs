//! OpenAI Chat Completions API types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------- Request ----------

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Not part of the OpenAI spec, but LiteLLM forwards it when Anthropic
    /// requests carry it. Most OpenAI-compatible servers ignore it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Anthropic-native name for stop. LiteLLM forwards this field unchanged
    /// when the downstream supports it; we keep both so callers can choose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// OpenAI uses the same field name `service_tier` ("auto" | "default" |
    /// "flex" | "scale" | "priority"). Anthropic's values are "auto" |
    /// "standard_only" — we map `standard_only` → `default` and pass
    /// `auto` through unchanged. Unknown values are forwarded verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum OpenAIMessage {
    System {
        content: OpenAIContent,
    },
    User {
        content: OpenAIContent,
    },
    Assistant {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<OpenAIContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<OpenAIToolCall>>,
        /// Concatenated text of any Anthropic `thinking` blocks attached to
        /// this assistant turn. Emitted by the `ReasoningContent` passthrough
        /// (DeepSeek / SGLang convention; vLLM aliases it to `reasoning`).
        /// Most other upstreams silently drop the unknown field. This is the
        /// only spelling with any real upstream consumer (Qwen3 chat template
        /// reads it).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        /// LiteLLM-internal shape: an array of structured thinking blocks
        /// preserved verbatim from the Anthropic input. LiteLLM itself
        /// strips this in its downstream provider transformations before
        /// the request goes on the wire, so no real upstream consumes it.
        /// Emitted only by the `LiteLLMThinkingBlocks` passthrough, used as
        /// a drop-in replacement for LiteLLM's intermediate adapter output.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking_blocks: Option<Vec<serde_json::Value>>,
    },
    Tool {
        tool_call_id: String,
        content: OpenAIContent,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAIImageUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: OpenAIFunctionDef,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIFunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIToolCall {
    /// Set only on the FIRST chunk of a streaming tool_call (per the OpenAI
    /// streaming contract). Continuation chunks may omit it (vLLM) or send
    /// it as explicit `null` (SGLang). Non-streaming tool_calls always
    /// carry an id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default = "default_function_type", skip_serializing_if = "String::is_empty")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
    /// Only present in streaming deltas; OpenAI uses this index to correlate
    /// streamed fragments to the same tool_call across chunks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIFunctionCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------- Response ----------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIChoice {
    pub index: i32,
    pub message: OpenAIChoiceMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIChoiceMessage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// DeepSeek / SGLang convention is `reasoning_content`; vLLM uses
    /// `reasoning`. We accept either on input and serialize as
    /// `reasoning_content`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning"
    )]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAIUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAIPromptTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

// ---------- Streaming chunk ----------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIStreamChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<OpenAIStreamChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAIStreamChoice {
    #[serde(default)]
    pub index: i32,
    pub delta: OpenAIStreamDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAIStreamDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}
