# Usage

Three ways to use cc_convert, plus testing and development workflow.

> Chinese docs: [USAGE.zh-CN.md](USAGE.zh-CN.md)

## 1. As a Python library

```bash
pip install cc_convert
```

```python
import cc_convert

# Translate an Anthropic-shape request to OpenAI shape
anthropic_req = {
    "model": "claude-opus-4-7",
    "max_tokens": 1000,
    "system": "You are a coding assistant.",
    "tools": [{
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
    }],
    "messages": [{"role":"user","content":"Read /etc/hosts"}],
}
openai_req, tool_map = cc_convert.translate_request(anthropic_req)
# POST openai_req to any OAI-compatible /v1/chat/completions endpoint

# Translate the upstream response back to Anthropic shape
openai_resp = {...}   # from your upstream
anthropic_resp = cc_convert.translate_response(
    openai_resp,
    original_model="claude-opus-4-7",
    tool_name_map=tool_map
)

# Streaming
translator = cc_convert.StreamTranslator("claude-opus-4-7", tool_map)
for openai_chunk in upstream_sse_stream:        # each is a dict
    for anthropic_event in translator.push(openai_chunk):
        # dict, type ∈ {message_start, ping, content_block_start,
        # content_block_delta, content_block_stop, message_delta, message_stop}
        emit_to_client(anthropic_event)
for trailing in translator.finish():
    emit_to_client(trailing)
```

### Two translation profiles

```python
# Pragmatic (default) — matches what real OAI-compat upstreams (vLLM/SGLang
# strict mode) actually accept:
#   - single-text content collapsed to string (many upstreams reject list-content)
#   - reasoning_effort auto-bucketed from thinking.budget_tokens
#   - max_completion_tokens for o1/o3/o4/gpt-5
#   - stream_options.include_usage auto-injected
cc_convert.translate_request(req)
cc_convert.translate_request(req, mode="pragmatic")

# LiteLLM byte-equivalent (drop-in replacement for LiteLLM AnthropicAdapter)
cc_convert.translate_request(req, mode="litellm_compat")
```

## 2. As a CLI sidecar (HTTP reverse proxy)

```bash
pip install cc_convert    # installs the `cc_convert` command

# Proxy mode: accept Anthropic-shape requests, forward to an OAI backend,
# translate the response back to Anthropic shape.
cc_convert serve \
    --listen 0.0.0.0:8787 \
    --upstream-url http://YOUR_UPSTREAM_HOST:8000 \
    --upstream-key sk-xxx                       # optional for local backends

# Then point any Anthropic-API client at it:
export ANTHROPIC_BASE_URL=http://localhost:8787
claude   # Claude Code thinks it's talking to Anthropic
```

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--mode {proxy,rpc}` | `proxy` | proxy forwards; rpc is translation-only |
| `--listen HOST:PORT` | `0.0.0.0:8787` | bind address |
| `--upstream-url URL` | `$CC_CONVERT_UPSTREAM_URL` | backend URL (auto-appends `/v1/chat/completions`) |
| `--upstream-key KEY` | `$CC_CONVERT_UPSTREAM_API_KEY` | bearer token sent to upstream |
| `--auth-passthrough` | off | forward client's Authorization header instead of `--upstream-key` |
| `--compat-mode {pragmatic,litellm_compat}` | `pragmatic` | translation profile |
| `--log-level {debug,info,warning,error}` | `info` | log level |
| `--log-format {text,json}` | `text` | json is one-object-per-line for log shipping |
| `-v` / `-vv` | | shorthand for `--log-level info/debug` |
| `--quiet` | off | suppress per-request access logs |
| `--version` | | print version |

Accepted endpoint paths: `/v1/messages`, `/messages`, `/anthropic/v1/messages` — anything ending in `/messages` or `/v1/messages` is recognised.

### One-shot CLI translation (no server)

```bash
# Anthropic request → OAI request
cat anthropic_req.json | cc_convert translate --direction cc-to-oai

# OAI response → Anthropic response
cat openai_resp.json | cc_convert translate \
    --direction oai-to-cc \
    --original-model claude-opus-4-7
```

### RPC mode (pure translation, no forwarding)

```bash
cc_convert serve --mode rpc --listen 127.0.0.1:8788

curl -X POST http://127.0.0.1:8788/translate/cc-to-oai -d '{...anthropic request...}'
curl -X POST http://127.0.0.1:8788/translate/oai-to-cc -d '{"openai_response":{...}, "original_model":"...", "tool_map":{}}'
```

## 3. As a pure Rust binary / library

```bash
cargo build --release -p cc_convert_sidecar     # static binary, no Python

export CC_CONVERT_UPSTREAM_URL="http://YOUR_UPSTREAM_HOST:8000/v1/chat/completions"
export CC_CONVERT_UPSTREAM_API_KEY="sk-..."
./target/release/cc_convert_sidecar
```

Same env-var contract as the Python CLI (all `CC_CONVERT_*` prefixed).

---

## What the upstream needs

cc_convert does NOT do client-side fallback parsing. If the upstream leaves
`<think>` or `<tool_call>` tags inside `content`, we faithfully pass them
through. The **real fix is on the upstream**:

### SGLang launch flags

| Symptom | Add this flag |
|---|---|
| `<think>...</think>` in content, `reasoning_content: null` | `--reasoning-parser qwen3` (or `deepseek-r1`, `hunyuan`, etc.) |
| `<tool_call>...</tool_call>` in content, `tool_calls: null` | `--tool-call-parser qwen25` (or `hermes`, `pythonic`, etc.) |

#### reasoning-parser options

`deepseek-r1` `deepseek-v3` `deepseek-v4` `qwen3` `qwen3-thinking` `glm45` `hunyuan` `gpt-oss` `kimi` `kimi_k2` `mistral` `mimo` `poolside_v1` `minimax` `minimax-append-think` `step3` `step3p5` `interns1` `nemotron_3` `gemma4`

#### tool-call-parser options

`qwen25` `qwen` `qwen3_coder` `hermes` `deepseekv3` `deepseekv31` `deepseekv32` `deepseekv4` `llama3` `mistral` `kimi_k2` `glm` `glm45` `glm47` `pythonic` `gpt-oss` `cohere_command4` `lfm2` `minicpm5` `mimo` `step3` `step3p5` `minimax-m2` `trinity` `interns1` `hunyuan` `gigachat3` `gemma4`

Example launch:

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen3-model \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen25 \
    ...
```

---

## Testing & development

### Run Rust tests

```bash
cargo test --workspace        # all (63 core + 3 sidecar)
cargo test -p cc_convert_core --tests
cargo test -p cc_convert_sidecar --test integration
```

### Run Python tests

```bash
cd python
maturin build --release       # produces ../target/wheels/cc_convert-*.whl
pip install --force-reinstall ../target/wheels/cc_convert-*.whl
pytest tests/                 # 69 tests
```

### Fast iteration on a change

```bash
cargo check -p cc_convert_core         # ~5s type-check, use this while editing
cargo test -p cc_convert_core --tests --lib   # ~30s, runs unit + rule tests
```

### Live round-trip against a real upstream

`playground/run_roundtrip.py` is the canonical end-to-end test:

```bash
python playground/run_roundtrip.py \
    --upstream http://YOUR_UPSTREAM_HOST:8000 \
    --model /model

# Outputs: playground/runs/<UTC-timestamp>/<fixture_name>/
#   1_anthropic_request.json    source Anthropic request (verbatim)
#   2_oai_request.json          what cc_convert sent to the upstream
#   3_oai_response.json         raw upstream response
#   4_anthropic_response.json   translated back to Anthropic
#   meta.json                   status / latency / http_status
#
# Plus _summary.json at the run root.
```

Run a single fixture:

```bash
python playground/run_roundtrip.py --upstream http://... --model /model --only agent_loop
```

### What each fixture exercises

| Fixture | Tests |
|---|---|
| `02_reasoning_request` | extended thinking; `thinking.budget_tokens` → `reasoning_effort` |
| `03_forced_tool` | `tool_choice:any` → OpenAI `required` |
| `05_simple_text` | single-turn baseline |
| `07_multi_turn_text` | 5-turn pure-text history |
| `08_agent_loop_with_tools` | **5-turn agent loop**: assistant calls 2 tools → user returns 2 tool_results → model continues |
| `09_long_response` | 4000-token long generation, large output + long latency |
| `10_parallel_tools_text_only` | one request triggers multiple parallel tool_use |

### Pre-push secret audit (recommended)

```bash
grep -rIEn "httpproxy|/workspace/|/root/|sk-[a-zA-Z0-9]{10,}|172\.27|10\.180" \
    --exclude-dir=target --exclude-dir=__pycache__ --exclude-dir=.git \
    --exclude-dir=playground/runs \
    --include="*.rs" --include="*.py" --include="*.toml" --include="*.md" .
```

`playground/runs/` is in .gitignore — live test results never enter git.

---

## FAQ

**Q: I see `reasoning_content: null` but `<think>` is still in content.**
A: Upstream hasn't enabled `--reasoning-parser`. Ask the operator to add it. cc_convert intentionally doesn't do client-side fallback.

**Q: Same for `tool_calls: null` but `<tool_call>` in content?**
A: Same — upstream needs `--tool-call-parser qwen25` (or `hermes`).

**Q: 503 / connection refused — is this cc_convert?**
A: No. Check `playground/runs/<ts>/<fx>/error.txt`. "No available workers" or "Connection refused" means upstream worker is unhealthy — our request is already on the wire and well-formed.

**Q: Does cc_convert drop Claude Code / OpenCode extra fields like `output_config`, `speed`, `container`?**
A: No (since commit `0be4d80`). All unknown fields are preserved in `AnthropicRequest.extra`. `output_config.effort` and `service_tier` are actually translated to their OpenAI equivalents; the others are kept for future Anthropic-target proxy mode.

**Q: Is the default LiteLLM-compatible?**
A: No. Default is `pragmatic` — matches what real OAI upstreams accept. Use `mode="litellm_compat"` (Python) or `--compat-mode litellm_compat` (CLI) for byte-parity with LiteLLM's AnthropicAdapter.
