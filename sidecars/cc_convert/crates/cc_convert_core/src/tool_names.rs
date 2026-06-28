//! OpenAI hard-limits tool names to 64 characters and constrains them to
//! `[a-zA-Z0-9_-]`. Anthropic does not. When an Anthropic tool name is too
//! long, we hash-truncate it to `{55-prefix}_{8-hex-sha}` and remember the
//! reverse mapping so we can restore the original name on the response side.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

const MAX_TOOL_NAME_LEN: usize = 64;

/// Round-trippable map from translated (≤64-char) OpenAI tool name to the
/// original Anthropic tool name. Names that don't need truncation are also
/// stored (mapped to themselves) so the response side can look up uniformly.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct ToolNameMap(pub HashMap<String, String>);

impl ToolNameMap {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Translate `original` to an OpenAI-safe name. If truncation is needed,
    /// remember the original→translated mapping (so the response side can
    /// restore it). Short names that did not need translation are NOT
    /// inserted into the map — matches LiteLLM behaviour.
    pub fn translate(&mut self, original: &str) -> String {
        if original.len() <= MAX_TOOL_NAME_LEN {
            return original.to_string();
        }
        let mut hasher = Sha256::new();
        hasher.update(original.as_bytes());
        let hash = hex::encode(hasher.finalize());
        let prefix: String = original.chars().take(55).collect();
        let safe = format!("{}_{}", prefix, &hash[..8]);
        self.0.insert(safe.clone(), original.to_string());
        safe
    }

    /// Look up the original name for a translated name. Falls back to the
    /// translated name itself if not registered (so unknown tool_calls are
    /// passed through unchanged).
    pub fn restore<'a>(&'a self, translated: &'a str) -> &'a str {
        self.0
            .get(translated)
            .map(String::as_str)
            .unwrap_or(translated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_names_passthrough() {
        let mut m = ToolNameMap::new();
        let s = m.translate("get_weather");
        assert_eq!(s, "get_weather");
        // Restore works without an explicit entry (fallback).
        assert_eq!(m.restore("get_weather"), "get_weather");
        // Map stays empty for short names, matching LiteLLM.
        assert!(m.0.is_empty());
    }

    #[test]
    fn long_names_truncated_and_round_trip() {
        let mut m = ToolNameMap::new();
        let long = "a".repeat(100);
        let s = m.translate(&long);
        assert!(s.len() <= 64);
        assert_eq!(m.restore(&s), long);
    }

    #[test]
    fn unknown_translated_name_passes_through() {
        let m = ToolNameMap::new();
        assert_eq!(m.restore("some_unknown"), "some_unknown");
    }
}
