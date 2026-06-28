//! cc_convert_core: bidirectional translator between the Anthropic Messages API
//! and the OpenAI Chat Completions API.
//!
//! Pure-Rust, no I/O. Used by both the Python wheel (`cc_convert_py`) and the
//! sidecar HTTP server (`cc_convert_sidecar`).

pub mod anthropic;
pub mod error;
pub mod openai;
pub mod req_to_openai;
pub mod resp_to_anthropic;
pub mod stream;
pub mod tool_names;

pub use error::ConvertError;
pub use req_to_openai::{anthropic_request_to_openai, ConvertOptions, ReasoningPassthrough};
pub use resp_to_anthropic::{
    openai_response_to_anthropic, openai_response_to_anthropic_with, ResponseConvertOptions,
};
pub use stream::{StreamConvertOptions, StreamTranslator};
pub use tool_names::ToolNameMap;

/// One-shot JSON-in / JSON-out request conversion. Returns
/// `{"openai_request": <OpenAIRequest>, "tool_map": <ToolNameMap>}`.
pub fn convert_request_json(input: &str, opts: &ConvertOptions) -> Result<String, ConvertError> {
    let req: anthropic::AnthropicRequest = serde_json::from_str(input)?;
    let (openai_req, tool_map) = anthropic_request_to_openai(&req, opts)?;
    let out = serde_json::json!({
        "openai_request": openai_req,
        "tool_map": tool_map,
    });
    Ok(serde_json::to_string(&out)?)
}

/// One-shot JSON-in / JSON-out response conversion.
pub fn convert_response_json(
    input: &str,
    original_model: &str,
    tool_map_json: &str,
) -> Result<String, ConvertError> {
    let resp: openai::OpenAIResponse = serde_json::from_str(input)?;
    let tool_map: ToolNameMap = serde_json::from_str(tool_map_json)?;
    let anthropic_resp = openai_response_to_anthropic(&resp, original_model, &tool_map)?;
    Ok(serde_json::to_string(&anthropic_resp)?)
}
