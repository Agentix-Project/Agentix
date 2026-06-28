//! Golden-file parity test: each input under tests/fixtures/requests/
//! anthropic_*.json is fed through the Rust translator under the
//! `litellm_compat` preset and the result is compared SEMANTICALLY against
//! the golden openai_*.json produced by LiteLLM.
//!
//! "Semantic" means: parse both sides into serde_json::Value, recursively
//! drop nulls, sort object keys, and compare. LiteLLM emits some fields
//! explicitly as null (`thinking_blocks: null`) that we omit; equivalent
//! shapes still pass.

use cc_convert_core::anthropic::AnthropicRequest;
use cc_convert_core::{anthropic_request_to_openai, ConvertOptions};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // workspace root
        .join("tests")
        .join("fixtures")
}

/// Recursively normalize JSON: drop nulls, recurse into objects/arrays.
/// Also normalize tool-call `arguments` (JSON-as-string) by reparsing and
/// re-stringifying with a stable separator-free format.
fn normalize(v: &Value, ctx_key: Option<&str>) -> Value {
    match v {
        Value::Null => Value::Null,
        Value::Bool(_) | Value::Number(_) => v.clone(),
        Value::String(s) => {
            if ctx_key == Some("arguments") {
                // Parse-and-restringify so whitespace differences don't matter.
                if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                    return Value::String(serde_json::to_string(&parsed).unwrap_or_default());
                }
            }
            Value::String(s.clone())
        }
        Value::Array(items) => Value::Array(items.iter().map(|i| normalize(i, None)).collect()),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, val) in map {
                let normalized = normalize(val, Some(k.as_str()));
                if matches!(normalized, Value::Null) {
                    continue;
                }
                out.insert(k.clone(), normalized);
            }
            Value::Object(out)
        }
    }
}

fn load_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {}", path.display(), e));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("parse {}: {}", path.display(), e))
}

fn collect_inputs(dir: &Path, prefix: &str) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read fixtures dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(rest) = name.strip_prefix(prefix) {
            out.push((rest.to_string(), path));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[test]
fn parity_with_litellm_request_fixtures() {
    let fixtures = fixtures_root().join("requests");
    let inputs = collect_inputs(&fixtures, "anthropic_");
    assert!(!inputs.is_empty(), "no fixtures under {}", fixtures.display());

    let mut failures = Vec::<String>::new();
    let mut checked = 0usize;
    let mut missing_golden = 0usize;

    for (name, input_path) in inputs {
        let golden_path = fixtures.join(format!("openai_{}.json", name));
        if !golden_path.exists() {
            missing_golden += 1;
            eprintln!("[skip] {name}: no golden ({})", golden_path.display());
            continue;
        }
        let anthropic_value = load_json(&input_path);
        let anthropic_req: AnthropicRequest = serde_json::from_value(anthropic_value)
            .unwrap_or_else(|e| panic!("parse fixture {name}: {e}"));

        let (openai_req, _tool_map) =
            anthropic_request_to_openai(&anthropic_req, &ConvertOptions::litellm_compat())
                .unwrap_or_else(|e| panic!("translate {name}: {e}"));

        let actual = normalize(&serde_json::to_value(&openai_req).unwrap(), None);
        let golden = normalize(&load_json(&golden_path), None);

        if actual != golden {
            failures.push(format!(
                "case {name}:\n  expected: {}\n    actual: {}\n",
                serde_json::to_string_pretty(&golden).unwrap(),
                serde_json::to_string_pretty(&actual).unwrap()
            ));
        }
        checked += 1;
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} request fixtures failed parity with LiteLLM:\n\n{}",
            failures.len(),
            checked,
            failures.join("\n---\n")
        );
    }

    eprintln!(
        "parity OK: {checked} request fixtures matched LiteLLM ({missing_golden} missing goldens)"
    );
}

#[test]
fn parity_tool_name_map_matches_litellm() {
    let fixtures = fixtures_root().join("requests");
    let inputs = collect_inputs(&fixtures, "anthropic_");
    let mut checked = 0;
    let mut failures = Vec::<String>::new();

    for (name, input_path) in inputs {
        let golden_map_path = fixtures.join(format!("tool_map_{}.json", name));
        if !golden_map_path.exists() {
            continue;
        }
        let anthropic_value = load_json(&input_path);
        let anthropic_req: AnthropicRequest = serde_json::from_value(anthropic_value).unwrap();
        let (_req, tool_map) =
            anthropic_request_to_openai(&anthropic_req, &ConvertOptions::litellm_compat())
                .unwrap();
        let actual = normalize(&serde_json::to_value(&tool_map).unwrap(), None);
        let golden = normalize(&load_json(&golden_map_path), None);
        if actual != golden {
            failures.push(format!(
                "case {name} tool_map:\n  expected: {}\n    actual: {}",
                serde_json::to_string(&golden).unwrap(),
                serde_json::to_string(&actual).unwrap()
            ));
        }
        checked += 1;
    }

    if !failures.is_empty() {
        panic!(
            "{} tool-map fixtures failed parity:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    eprintln!("parity OK: {checked} tool-map fixtures matched LiteLLM");
}
