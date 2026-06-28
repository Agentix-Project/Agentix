//! HTTP sidecar logic, factored into a library so integration tests can
//! reuse [`AppState`] and [`build_router`].

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use cc_convert_core::{
    anthropic::{AnthropicEvent, AnthropicRequest, AnthropicResponse},
    anthropic_request_to_openai,
    openai::OpenAIStreamChunk,
    openai_response_to_anthropic, ConvertOptions, StreamConvertOptions, StreamTranslator,
};
use futures::stream::{self, Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::{collections::VecDeque, sync::Arc};

#[derive(Clone)]
pub struct AppState {
    pub upstream_url: String,
    pub upstream_key: Option<String>,
    pub auth_passthrough: bool,
    pub http: Client,
    /// If true, use ConvertOptions::litellm_compat() (preserve LiteLLM-equivalent
    /// behaviour). Default false → ConvertOptions::pragmatic() (collapses
    /// single-text content into a string, which is what most real upstreams
    /// expect — SGLang/vLLM strict mode rejects list-content on system msgs).
    pub litellm_compat: bool,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/v1/messages", post(handle_messages))
        .with_state(state)
}

fn convert_options_for(state: &AppState) -> ConvertOptions {
    if state.litellm_compat {
        ConvertOptions::litellm_compat()
    } else {
        ConvertOptions::pragmatic()
    }
}

pub async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req_value): Json<Value>,
) -> Response {
    let stream_mode = req_value
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let original_model = req_value
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let anthropic_req: AnthropicRequest = match serde_json::from_value(req_value) {
        Ok(r) => r,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_request_error", &e.to_string())
        }
    };

    let (openai_req, tool_map) =
        match anthropic_request_to_openai(&anthropic_req, &convert_options_for(&state)) {
            Ok(p) => p,
            Err(e) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    &e.to_string(),
                )
            }
        };

    let auth_header = if state.auth_passthrough {
        headers
            .get("authorization")
            .or_else(|| headers.get("x-api-key"))
            .and_then(|v| v.to_str().ok())
            .map(|s| {
                if s.starts_with("Bearer ") {
                    s.to_string()
                } else {
                    format!("Bearer {}", s)
                }
            })
    } else {
        state.upstream_key.as_ref().map(|k| format!("Bearer {}", k))
    };

    let mut req_builder = state
        .http
        .post(&state.upstream_url)
        .json(&openai_req)
        .header("content-type", "application/json");
    if let Some(auth) = &auth_header {
        req_builder = req_builder.header("authorization", auth);
    }

    let upstream_resp = match req_builder.send().await {
        Ok(r) => r,
        Err(e) => {
            return error_response(StatusCode::BAD_GATEWAY, "api_error", &e.to_string());
        }
    };

    let status = upstream_resp.status();
    if !status.is_success() {
        let body = upstream_resp
            .text()
            .await
            .unwrap_or_else(|_| "upstream error".to_string());
        return error_response(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            "api_error",
            &body,
        );
    }

    if !stream_mode {
        let resp_value: Value = match upstream_resp.json().await {
            Ok(v) => v,
            Err(e) => return error_response(StatusCode::BAD_GATEWAY, "api_error", &e.to_string()),
        };
        let openai_resp = match serde_json::from_value(resp_value) {
            Ok(r) => r,
            Err(e) => return error_response(StatusCode::BAD_GATEWAY, "api_error", &e.to_string()),
        };
        let anthropic_resp: AnthropicResponse =
            match openai_response_to_anthropic(&openai_resp, &original_model, &tool_map) {
                Ok(r) => r,
                Err(e) => {
                    return error_response(StatusCode::BAD_GATEWAY, "api_error", &e.to_string())
                }
            };
        return (StatusCode::OK, Json(anthropic_resp)).into_response();
    }

    let translator = StreamTranslator::with_options(
        original_model,
        tool_map,
        StreamConvertOptions::anthropic_native(),
    );
    let upstream = upstream_resp.bytes_stream();
    let event_stream = build_sse_stream(translator, upstream);
    Sse::new(event_stream).into_response()
}

fn error_response(status: StatusCode, type_: &str, message: &str) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": { "type": type_, "message": message }
    });
    (status, Json(body)).into_response()
}

struct SseState<S> {
    translator: StreamTranslator,
    upstream: S,
    buffer: Vec<u8>,
    queued: VecDeque<AnthropicEvent>,
    upstream_done: bool,
    finalized: bool,
}

pub fn build_sse_stream<S>(
    translator: StreamTranslator,
    upstream: S,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + Unpin + 'static,
{
    let init = SseState {
        translator,
        upstream,
        buffer: Vec::new(),
        queued: VecDeque::new(),
        upstream_done: false,
        finalized: false,
    };
    stream::unfold(init, |mut st| async move {
        loop {
            if let Some(ev) = st.queued.pop_front() {
                return Some((Ok(anthropic_event_to_sse(&ev)), st));
            }
            if st.finalized {
                return None;
            }
            if st.upstream_done {
                drain_buffer(&mut st);
                st.queued.extend(st.translator.finish());
                st.finalized = true;
                continue;
            }
            match st.upstream.next().await {
                Some(Ok(bytes)) => {
                    st.buffer.extend_from_slice(&bytes);
                    drain_buffer(&mut st);
                }
                Some(Err(_)) | None => {
                    st.upstream_done = true;
                }
            }
        }
    })
}

fn drain_buffer<S>(st: &mut SseState<S>) {
    loop {
        let Some(sep_pos) = st.buffer.windows(2).position(|w| w == b"\n\n") else {
            break;
        };
        let event_bytes: Vec<u8> = st.buffer.drain(..sep_pos).collect();
        st.buffer.drain(..2);
        let Ok(event_str) = std::str::from_utf8(&event_bytes) else {
            continue;
        };
        for line in event_str.lines() {
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let payload = data.trim();
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(payload) else {
                continue;
            };
            let events = st.translator.push_openai_chunk(&chunk);
            st.queued.extend(events);
        }
    }
}

fn anthropic_event_to_sse(ev: &AnthropicEvent) -> Event {
    let (name, value) = (event_name(ev), serde_json::to_string(ev).unwrap_or_default());
    Event::default().event(name).data(value)
}

fn event_name(ev: &AnthropicEvent) -> &'static str {
    match ev {
        AnthropicEvent::MessageStart { .. } => "message_start",
        AnthropicEvent::Ping => "ping",
        AnthropicEvent::ContentBlockStart { .. } => "content_block_start",
        AnthropicEvent::ContentBlockDelta { .. } => "content_block_delta",
        AnthropicEvent::ContentBlockStop { .. } => "content_block_stop",
        AnthropicEvent::MessageDelta { .. } => "message_delta",
        AnthropicEvent::MessageStop => "message_stop",
    }
}
