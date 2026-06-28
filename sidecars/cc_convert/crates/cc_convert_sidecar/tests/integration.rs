//! Integration test: spin up a mock OpenAI-compatible upstream + the sidecar
//! proxy, send an Anthropic-shape request through the proxy, and verify the
//! Anthropic-shape response.
//!
//! Covers:
//!   1. Non-streaming round-trip (request translated + forwarded, response
//!      translated back).
//!   2. Streaming round-trip (OpenAI SSE → Anthropic SSE).
//!   3. Upstream 4xx propagated as Anthropic-shape error JSON.

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::net::TcpListener;

#[derive(Default)]
struct MockState {
    last_request: Mutex<Option<Value>>,
    last_auth: Mutex<Option<String>>,
    mode: Mutex<MockMode>,
}

#[derive(Default, Clone, Copy)]
enum MockMode {
    #[default]
    NonStreaming,
    Streaming,
    Failure4xx,
}

async fn mock_handler(
    State(state): State<Arc<MockState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    *state.last_request.lock().unwrap() = Some(body.clone());
    *state.last_auth.lock().unwrap() = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let mode = *state.mode.lock().unwrap();
    match mode {
        MockMode::NonStreaming => Json(json!({
            "id": "chatcmpl-abc",
            "model": body.get("model").cloned().unwrap_or(json!("gpt-4o-mini")),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi from upstream"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3}
        }))
        .into_response(),
        MockMode::Streaming => {
            let chunks = vec![
                "data: {\"id\":\"chatcmpl-s\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hel\"}}]}\n\n".to_string(),
                "data: {\"id\":\"chatcmpl-s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n".to_string(),
                "data: {\"id\":\"chatcmpl-s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string(),
                "data: [DONE]\n\n".to_string(),
            ];
            let body = Body::from_stream(futures::stream::iter(
                chunks
                    .into_iter()
                    .map(|c| Ok::<_, std::convert::Infallible>(c.into_bytes())),
            ));
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                body,
            )
                .into_response()
        }
        MockMode::Failure4xx => (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": "bad upstream request", "type": "invalid_request_error"}})),
        )
            .into_response(),
    }
}

async fn spawn_mock() -> (Arc<MockState>, SocketAddr) {
    let state = Arc::new(MockState::default());
    let app = Router::new()
        .route("/v1/chat/completions", post(mock_handler))
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (state, addr)
}

async fn spawn_sidecar(upstream_url: String, upstream_key: Option<String>) -> SocketAddr {
    use cc_convert_sidecar::*; // re-exported router builder

    let state = AppState {
        upstream_url,
        upstream_key,
        auth_passthrough: false,
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap(),
        litellm_compat: false,
    };
    let app = build_router(std::sync::Arc::new(state));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

#[tokio::test]
async fn non_streaming_round_trip() {
    let (mock_state, mock_addr) = spawn_mock().await;
    *mock_state.mode.lock().unwrap() = MockMode::NonStreaming;

    let upstream_url = format!("http://{}/v1/chat/completions", mock_addr);
    let sidecar_addr = spawn_sidecar(upstream_url, Some("k".to_string())).await;

    let client = reqwest::Client::new();
    let resp: Value = client
        .post(format!("http://{}/v1/messages", sidecar_addr))
        .json(&json!({
            "model": "claude-opus-4-7",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "ping"}]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    // Anthropic-shape response.
    assert_eq!(resp["model"], "claude-opus-4-7");
    assert_eq!(resp["role"], "assistant");
    assert_eq!(resp["type"], "message");
    assert_eq!(resp["content"][0]["text"], "hi from upstream");
    assert_eq!(resp["stop_reason"], "end_turn");

    // Upstream saw the OpenAI-shape request.
    let upstream_req = mock_state.last_request.lock().unwrap().clone().unwrap();
    assert_eq!(upstream_req["messages"][0]["role"], "user");
    assert_eq!(upstream_req["messages"][0]["content"], "ping");
    assert_eq!(upstream_req["max_tokens"], 100);

    // Auth header carried the configured key.
    assert_eq!(
        mock_state.last_auth.lock().unwrap().as_deref(),
        Some("Bearer k")
    );
}

#[tokio::test]
async fn streaming_round_trip() {
    let (mock_state, mock_addr) = spawn_mock().await;
    *mock_state.mode.lock().unwrap() = MockMode::Streaming;

    let upstream_url = format!("http://{}/v1/chat/completions", mock_addr);
    let sidecar_addr = spawn_sidecar(upstream_url, Some("k".to_string())).await;

    let client = reqwest::Client::new();
    let mut resp = client
        .post(format!("http://{}/v1/messages", sidecar_addr))
        .json(&json!({
            "model": "claude-opus-4-7",
            "max_tokens": 50,
            "stream": true,
            "messages": [{"role": "user", "content": "ping"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.starts_with("text/event-stream"),
        "content-type was: {ct}"
    );

    let mut body = String::new();
    while let Some(chunk) = resp.chunk().await.unwrap() {
        body.push_str(std::str::from_utf8(&chunk).unwrap());
    }

    // Validate the Anthropic SSE event sequence by extracting `event:` lines.
    let event_names: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("event: "))
        .collect();
    assert!(event_names.contains(&"message_start"), "events: {event_names:?}");
    assert!(event_names.contains(&"ping"), "events: {event_names:?}");
    assert!(event_names.contains(&"content_block_start"));
    assert!(event_names.contains(&"content_block_delta"));
    assert!(event_names.contains(&"content_block_stop"));
    assert!(event_names.contains(&"message_delta"));
    assert!(event_names.contains(&"message_stop"));

    // The concatenated text_delta payloads should reconstruct "hello".
    let mut text = String::new();
    for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            if v["type"] == "content_block_delta"
                && v["delta"]["type"] == "text_delta"
            {
                if let Some(s) = v["delta"]["text"].as_str() {
                    text.push_str(s);
                }
            }
        }
    }
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn upstream_4xx_surfaces_anthropic_error_shape() {
    let (mock_state, mock_addr) = spawn_mock().await;
    *mock_state.mode.lock().unwrap() = MockMode::Failure4xx;

    let upstream_url = format!("http://{}/v1/chat/completions", mock_addr);
    let sidecar_addr = spawn_sidecar(upstream_url, Some("k".to_string())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{}/v1/messages", sidecar_addr))
        .json(&json!({
            "model": "claude-opus-4-7",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "ping"}]
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body: Value = resp.json().await.unwrap();

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["type"], "error");
    assert!(body["error"]["message"].as_str().unwrap().contains("bad upstream request"));
}
