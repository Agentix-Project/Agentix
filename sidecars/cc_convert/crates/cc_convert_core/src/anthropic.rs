//! Anthropic Messages API types.
//!
//! Only the subset of fields we actively translate is modeled. Unknown fields
//! are preserved on request-shaped types via `extra` catch-alls where useful,
//! and dropped on response-shaped types (we emit a fixed surface).

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------- Request ----------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemField>,

    pub max_tokens: u32,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,

    /// Anthropic Messages API extension fields used by Claude Code, OpenCode,
    /// Cline, and the Anthropic SDK that aren't modeled individually here:
    /// `output_config`, `context_management`, `speed`, `container`,
    /// `mcp_servers`, `inference_geo`, `cache_control` (top-level),
    /// `service_tier`, `diagnostics`, `betas`, plus anything injected via
    /// `CLAUDE_CODE_EXTRA_BODY`. Captured here so we don't silently drop
    /// them — the request translator can choose to map known ones and pass
    /// the rest through when the upstream is itself /v1/messages-compatible.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SystemField {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String, // typically "text"
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String, // "user" | "assistant"
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    Image {
        source: ImageSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<Value>,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    /// Catch-all for Anthropic content blocks we don't model individually
    /// (server_tool_use, web_search_tool_result, code_execution_tool_result,
    /// bash_code_execution_tool_result, text_editor_code_execution_tool_result,
    /// tool_search_tool_result, mcp_tool_use, mcp_tool_result, container_upload,
    /// document, etc.). Translator drops these by default; the
    /// pass-through `Value` lets callers inspect them if needed.
    #[serde(other, skip_serializing)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
}

/// One tool definition in an Anthropic request.
///
/// Anthropic accepts two shapes here:
///
/// - **Client tools** (the common case): a free-form schema you've defined.
///   The wire shape is `{name, description?, input_schema, cache_control?}`
///   with either no `type` field or `type:"custom"`.
///
/// - **Server / hosted tools** (Anthropic-only): tools the Anthropic backend
///   itself executes — `{type:"web_search_20250305", name:"web_search", ...}`,
///   `{type:"computer_20241022", ...}`, `bash_*`, `text_editor_*`,
///   `web_fetch_*`, `code_execution_*`, `tool_search_*`. These have NO
///   `input_schema` and carry tool-version-specific config fields. There is
///   no OpenAI Chat Completions equivalent — OpenAI's tools array only
///   accepts `{type:"function", function:{...}}`, so a translator that
///   forwards these unchanged produces an HTTP 400 the moment the request
///   reaches any real OpenAI-compatible upstream.
///
/// The deserializer routes by presence of `input_schema`: if it's there,
/// the tool is a Client tool; otherwise it's Hosted. We keep the full raw
/// JSON of Hosted tools in `raw` so the translator can log what it dropped
/// (useful for users debugging "why didn't my web_search tool fire").
#[derive(Debug, Clone)]
pub enum AnthropicTool {
    Client(AnthropicClientTool),
    Hosted(AnthropicHostedTool),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicClientTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicHostedTool {
    /// e.g. "web_search_20250305", "computer_20241022", "bash_20250124", ...
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Anthropic-side name (e.g. "web_search"); not always present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Everything else (max_uses, user_location, allowed_domains, ...) kept
    /// as raw JSON. Lets the translator log what it dropped without modeling
    /// every per-version variant.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

impl<'de> Deserialize<'de> for AnthropicTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = Value::deserialize(deserializer)?;
        // Heuristic: a tool with `input_schema` is a client tool; anything else
        // (especially anything with a non-default `type`) is hosted.
        if v.get("input_schema").is_some() {
            let t: AnthropicClientTool =
                serde_json::from_value(v).map_err(serde::de::Error::custom)?;
            Ok(AnthropicTool::Client(t))
        } else {
            let t: AnthropicHostedTool =
                serde_json::from_value(v).map_err(serde::de::Error::custom)?;
            Ok(AnthropicTool::Hosted(t))
        }
    }
}

impl Serialize for AnthropicTool {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            AnthropicTool::Client(t) => t.serialize(serializer),
            AnthropicTool::Hosted(t) => t.serialize(serializer),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AnthropicMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Catch-all for any other metadata sub-fields some clients might
    /// invent (none are documented at the time of writing — Claude Code
    /// stuffs device_id/account_uuid/session_id inside the `user_id`
    /// STRING as serialized JSON rather than adding sibling keys, but
    /// keeping this open avoids silent drops if that ever changes).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicThinking {
    Enabled { budget_tokens: u32 },
    Disabled,
}

// ---------- Response ----------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String, // "message"
    pub role: String,     // "assistant"
    pub model: String,
    pub content: Vec<ResponseContentBlock>,
    pub stop_reason: AnthropicStopReason,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

// ---------- Streaming events (output of StreamTranslator) ----------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicEvent {
    MessageStart {
        message: MessageStartPayload,
    },
    Ping,
    ContentBlockStart {
        index: i32,
        content_block: StreamingContentBlock,
    },
    ContentBlockDelta {
        index: i32,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: i32,
    },
    MessageDelta {
        delta: MessageDeltaPayload,
        usage: AnthropicUsage,
    },
    MessageStop,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageStartPayload {
    pub id: String,
    #[serde(rename = "type")]
    pub msg_type: String, // "message"
    pub role: String,     // "assistant"
    pub model: String,
    pub content: Vec<ResponseContentBlock>, // always empty []
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamingContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
    Thinking { thinking: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageDeltaPayload {
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
}
