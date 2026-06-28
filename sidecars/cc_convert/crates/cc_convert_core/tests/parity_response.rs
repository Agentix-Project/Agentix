//! Response parity vs LiteLLM. For each
//! tests/fixtures/responses/openai_<name>.json (with sidecar
//! meta_<name>.json carrying any tool_map), feeds the input through
//! `openai_response_to_anthropic_with(... litellm_compat ...)` and asserts
//! semantic equality against the LiteLLM-produced golden
//! anthropic_<name>.json.

use cc_convert_core::{
    openai::OpenAIResponse, openai_response_to_anthropic_with, tool_names::ToolNameMap,
    ResponseConvertOptions,
};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("responses")
}

fn normalize(v: &Value) -> Value {
    match v {
        Value::Null => Value::Null,
        Value::Bool(_) | Value::Number(_) | Value::String(_) => v.clone(),
        Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, val) in map {
                let n = normalize(val);
                if matches!(n, Value::Null) {
                    continue;
                }
                out.insert(k.clone(), n);
            }
            Value::Object(out)
        }
    }
}

fn load(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn response_parity_with_litellm() {
    let root = fixtures_root();
    let mut failures = Vec::<String>::new();
    let mut checked = 0;
    let mut missing = 0;

    for entry in std::fs::read_dir(&root).expect("read responses fixtures dir") {
        let path = entry.unwrap().path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(name) = stem.strip_prefix("openai_") else {
            continue;
        };
        let golden_path = root.join(format!("anthropic_{name}.json"));
        if !golden_path.exists() {
            missing += 1;
            continue;
        }
        let meta_path = root.join(format!("meta_{name}.json"));
        let meta_val: Value = if meta_path.exists() {
            load(&meta_path)
        } else {
            Value::Object(Default::default())
        };
        let tool_map: ToolNameMap = meta_val
            .get("tool_map")
            .map(|v| serde_json::from_value(v.clone()).unwrap_or_default())
            .unwrap_or_default();

        let openai_resp: OpenAIResponse = serde_json::from_value(load(&path))
            .unwrap_or_else(|e| panic!("parse {name}: {e}"));

        let opts = ResponseConvertOptions::litellm_compat();
        let anthropic = openai_response_to_anthropic_with(&openai_resp, &tool_map, &opts)
            .unwrap_or_else(|e| panic!("translate {name}: {e}"));
        let actual = normalize(&serde_json::to_value(&anthropic).unwrap());
        let golden = normalize(&load(&golden_path));

        if actual != golden {
            failures.push(format!(
                "case {name}:\n  expected: {}\n    actual: {}",
                serde_json::to_string_pretty(&golden).unwrap(),
                serde_json::to_string_pretty(&actual).unwrap(),
            ));
        }
        checked += 1;
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} response fixtures diverged:\n\n{}",
            failures.len(),
            checked,
            failures.join("\n---\n")
        );
    }
    eprintln!("response parity OK: {checked} fixtures matched LiteLLM ({missing} missing)");
}
