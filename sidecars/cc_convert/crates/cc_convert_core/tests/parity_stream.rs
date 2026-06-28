//! Stream parity vs LiteLLM. For each
//! tests/fixtures/streams/openai_<name>.sse, parses the SSE into chunks,
//! feeds them through `StreamTranslator` (litellm_compat preset), and
//! asserts the resulting event sequence equals the LiteLLM-produced
//! anthropic_<name>.jsonl golden — modulo dropped nulls and the
//! non-deterministic message_start id.

use cc_convert_core::openai::OpenAIStreamChunk;
use cc_convert_core::tool_names::ToolNameMap;
use cc_convert_core::{StreamConvertOptions, StreamTranslator};
use serde_json::Value;
use std::path::PathBuf;

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("streams")
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

/// Mask the message_start.message.id since LiteLLM uses a random uuid each
/// time. Both sides become `"__masked__"` for comparison.
fn mask_message_id(events: &mut [Value]) {
    for ev in events.iter_mut() {
        if ev.get("type").and_then(|v| v.as_str()) == Some("message_start") {
            if let Some(msg) = ev.get_mut("message").and_then(|m| m.as_object_mut()) {
                if let Some(id) = msg.get_mut("id") {
                    *id = Value::String("__masked__".to_string());
                }
            }
        }
    }
}

fn parse_sse(text: &str) -> Vec<OpenAIStreamChunk> {
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        for line in block.lines() {
            if let Some(payload) = line.strip_prefix("data:") {
                let payload = payload.trim();
                if payload.is_empty() || payload == "[DONE]" {
                    continue;
                }
                if let Ok(c) = serde_json::from_str::<OpenAIStreamChunk>(payload) {
                    out.push(c);
                }
            }
        }
    }
    out
}

fn run_translator(chunks: &[OpenAIStreamChunk]) -> Vec<Value> {
    let mut t = StreamTranslator::with_options(
        "claude-opus-4-7".to_string(),
        ToolNameMap::new(),
        StreamConvertOptions::litellm_compat(),
    );
    let mut events = Vec::new();
    for c in chunks {
        for ev in t.push_openai_chunk(c) {
            events.push(serde_json::to_value(&ev).unwrap());
        }
    }
    for ev in t.finish() {
        events.push(serde_json::to_value(&ev).unwrap());
    }
    events
}

fn load_jsonl(path: &std::path::Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Cases where LiteLLM's AnthropicStreamWrapper emits per-spec-WRONG events
/// (documented quirks). We intentionally do not match these byte-for-byte
/// because our behaviour is closer to the Anthropic SSE spec. The
/// `stream_translation.rs` unit-test file verifies these cases work
/// correctly under our own semantics.
///
/// - `29_two_parallel_tool_calls`: LiteLLM merges both tool_calls into one
///   content block and concats their arguments (`"{}{}"`). Per spec each
///   parallel tool_call should be its own block.
/// - `30_stream_ends_without_finish_reason`: LiteLLM never emits the closing
///   `content_block_stop` / `message_delta` / `message_stop` events. We
///   emit them so downstream Anthropic clients aren't left hanging.
/// - `31_reasoning_then_text`: LiteLLM emits both thinking and text as
///   deltas to the same content_block at index 0. Per spec they should be
///   separate blocks (thinking + text).
const LITELLM_QUIRKS_TO_SKIP: &[&str] = &[
    "29_two_parallel_tool_calls",
    "30_stream_ends_without_finish_reason",
    "31_reasoning_then_text",
];

#[test]
fn stream_parity_with_litellm() {
    let root = fixtures_root();
    let mut failures = Vec::<String>::new();
    let mut checked = 0;
    let mut missing = 0;
    let mut skipped = 0;

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("read streams dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("sse")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with("openai_"))
                    .unwrap_or(false)
            {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    entries.sort();

    for path in entries {
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let name = stem.strip_prefix("openai_").unwrap();
        if LITELLM_QUIRKS_TO_SKIP.contains(&name) {
            skipped += 1;
            continue;
        }
        let golden_path = root.join(format!("anthropic_{name}.jsonl"));
        if !golden_path.exists() {
            missing += 1;
            continue;
        }
        let sse_text = std::fs::read_to_string(&path).unwrap();
        let chunks = parse_sse(&sse_text);
        let mut actual = run_translator(&chunks);
        let mut golden = load_jsonl(&golden_path);
        mask_message_id(&mut actual);
        mask_message_id(&mut golden);
        let actual_norm: Vec<Value> = actual.iter().map(normalize).collect();
        let golden_norm: Vec<Value> = golden.iter().map(normalize).collect();

        if actual_norm != golden_norm {
            failures.push(format!(
                "case {name}:\n  expected (LiteLLM): {}\n  actual (cc_convert): {}",
                serde_json::to_string_pretty(&Value::Array(golden_norm)).unwrap(),
                serde_json::to_string_pretty(&Value::Array(actual_norm)).unwrap(),
            ));
        }
        checked += 1;
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} stream fixtures diverged:\n\n{}",
            failures.len(),
            checked,
            failures.join("\n---\n")
        );
    }
    eprintln!(
        "stream parity OK: {checked} fixtures matched LiteLLM \
         ({skipped} skipped LiteLLM-quirks, {missing} missing goldens)"
    );
}
