use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConvertError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("unsupported feature: {0}")]
    Unsupported(String),
}
