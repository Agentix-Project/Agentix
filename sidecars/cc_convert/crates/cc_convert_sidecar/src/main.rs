//! Thin binary entry point. All logic lives in `cc_convert_sidecar::lib`.

use cc_convert_sidecar::{build_router, AppState};
use reqwest::Client;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tracing_subscriber::EnvFilter;

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let listen_addr = env_or_default("CC_CONVERT_LISTEN_ADDR", "0.0.0.0:8787");
    let upstream_url = env_or_default(
        "CC_CONVERT_UPSTREAM_URL",
        "https://api.openai.com/v1/chat/completions",
    );
    let upstream_key = std::env::var("CC_CONVERT_UPSTREAM_API_KEY").ok();
    let auth_passthrough = std::env::var("CC_CONVERT_AUTH_PASSTHROUGH")
        .map(|v| v == "1")
        .unwrap_or(false);
    let litellm_compat = std::env::var("CC_CONVERT_LITELLM_COMPAT")
        .map(|v| v == "1")
        .unwrap_or(false);

    let state = AppState {
        upstream_url,
        upstream_key,
        auth_passthrough,
        http: Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("reqwest client"),
        litellm_compat,
    };

    let app = build_router(Arc::new(state));

    let addr: SocketAddr = listen_addr.parse().expect("invalid CC_CONVERT_LISTEN_ADDR");
    tracing::info!(%addr, "cc_convert_sidecar listening");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}
