# cc_convert

> 中文文档:[README.zh-CN.md](README.zh-CN.md)
>
> 详细用法 / Usage details: [USAGE.md](USAGE.md) ([中文](USAGE.zh-CN.md))

Bidirectional Anthropic ↔ OpenAI Chat Completions protocol converter, written
in Rust. Lets clients keep calling the Anthropic Messages API shape (system
prompt + content blocks + `tool_use` / `tool_result` + Anthropic SSE) while
the actual upstream is an OpenAI-compatible server, and vice versa for
responses.

Validated against three reference implementations:
- [LiteLLM]'s `AnthropicAdapter` (primary parity oracle — request, response, stream)
- [1rgs/claude-code-proxy], [maxnowack/anthropic-proxy] (cross-reference)
- [THUDM/slime]'s anthropic adapter (Chinese RL ecosystem)

And hardened against the **non-standard quirks** of self-hosted servers:
- **vLLM**: uses `reasoning` (not `reasoning_content`); extra `stop_reason`,
  `prompt_logprobs`, `kv_transfer_params` fields; first stream chunk is
  role-only.
- **SGLang**: emits `id: null` and `function.name: null` on continuation
  tool_call chunks; sends `reasoning_content: null` on every chunk;
  `matched_stop`, `metadata`, `sglext` extras; `finish_reason: "abort"`;
  kimi_k2 tool IDs of form `functions.<name>:<idx>`.

[LiteLLM]: https://github.com/BerriAI/litellm
[1rgs/claude-code-proxy]: https://github.com/1rgs/claude-code-proxy
[maxnowack/anthropic-proxy]: https://github.com/maxnowack/anthropic-proxy
[THUDM/slime]: https://github.com/THUDM/slime/tree/main/slime/agent/adapters

## Two deployment modes

| Mode | What it is | Use when |
|---|---|---|
| Python package | `pip install cc_convert`, `import cc_convert` | You're embedding the converter in a Python app and prefer dict-in / dict-out. |
| HTTP sidecar | `cc_convert_sidecar` binary; listens on `/v1/messages` (Anthropic shape) and proxies to a configured OpenAI-compatible upstream | You're plugging an Anthropic-API client (Claude Code, claude-py, etc.) into a non-Anthropic backend. |

Both paths share the same Rust translation core (`cc_convert_core`), so
behaviour is identical.

## Python usage

```python
import cc_convert

# 1) Translate an Anthropic request → OpenAI request.
anthropic_req = {
    "model": "gpt-4o-mini",
    "max_tokens": 500,
    "system": "Be concise.",
    "tools": [{
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }],
    "tool_choice": {"type": "any"},
    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
}
openai_req, tool_map = cc_convert.translate_request(anthropic_req)
# → POST openai_req to your OpenAI-compatible /v1/chat/completions endpoint.

# 2) Translate the OpenAI response back to Anthropic shape.
openai_resp = {...}  # from your upstream
anthropic_resp = cc_convert.translate_response(
    openai_resp, original_model="claude-opus-4-7", tool_name_map=tool_map
)

# 3) Streaming: feed OpenAI SSE chunks, get Anthropic SSE events.
translator = cc_convert.StreamTranslator("claude-opus-4-7", tool_map)
for openai_chunk in upstream_sse_chunks:           # each is a dict
    for anthropic_event in translator.push(openai_chunk):
        emit_to_client(anthropic_event)            # type: message_start / content_block_* / message_delta / message_stop
for trailing in translator.finish():
    emit_to_client(trailing)
```

The first element of `translate_request`'s return value is the OpenAI body
you POST. The second is a `dict[str, str]` mapping translated → original
tool names — keep it for the response side so we can restore tool names that
were truncated to fit OpenAI's 64-char limit.

## Sidecar usage

You can run the sidecar two ways:

### A) Built-in CLI (Python wheel) — simplest

After `pip install cc_convert`, the wheel exposes a `cc_convert` command.

```bash
# Run the sidecar in proxy mode (Anthropic-shape in → OpenAI upstream → Anthropic-shape out).
cc_convert serve \
    --listen 0.0.0.0:8787 \
    --upstream-url https://api.openai.com/v1/chat/completions \
    --upstream-key sk-...

# Same thing pointing at a vLLM / SGLang / DeepSeek backend:
cc_convert serve --upstream-url http://localhost:8000/v1/chat/completions -v

# Pure-translation RPC server (no upstream call):
cc_convert serve --mode rpc --listen 127.0.0.1:8788

# One-shot JSON in / JSON out (no server):
cat anthropic_req.json | cc_convert translate --direction cc-to-oai
cat openai_resp.json  | cc_convert translate --direction oai-to-cc --original-model claude-opus-4-7
```

Useful flags:

| Flag | Default | What it does |
|---|---|---|
| `--mode {proxy,rpc}` | `proxy` | `proxy`: terminate Anthropic on `--cc-path`, forward to `--upstream-url`, return translated Anthropic. `rpc`: stateless `/translate/cc-to-oai` and `/translate/oai-to-cc` endpoints. |
| `--listen HOST:PORT` | `0.0.0.0:8787` | What to bind to. |
| `--upstream-url URL` | `$CC_CONVERT_UPSTREAM_URL` | (proxy) The OpenAI-compatible `/v1/chat/completions` URL. |
| `--upstream-key KEY` | `$CC_CONVERT_UPSTREAM_API_KEY` | (proxy) Bearer token sent to the upstream. |
| `--auth-passthrough` | off | Use the CLIENT's `Authorization` / `x-api-key` header instead of `--upstream-key`. |
| `--cc-path PATH` | `/v1/messages` | (proxy) Path that receives Anthropic-shape requests. |
| `--cc-to-oai-path PATH` | `/translate/cc-to-oai` | (rpc) Path for request-side translation. |
| `--oai-to-cc-path PATH` | `/translate/oai-to-cc` | (rpc) Path for response-side translation. |
| `--log-level LEVEL` | `info` | `debug` / `info` / `warning` / `error`. |
| `--log-format FORMAT` | `text` | `text` (human) or `json` (one JSON object per line, easy to ship). |
| `-v`, `-vv` | | Shorthand for `--log-level info` / `debug`. |
| `--quiet` | off | Suppress per-request access logs. |
| `--version` | | Print version and exit. |

All flags also accept env-var defaults: `CC_CONVERT_MODE`, `CC_CONVERT_LISTEN_ADDR`,
`CC_CONVERT_UPSTREAM_URL`, `CC_CONVERT_UPSTREAM_API_KEY`,
`CC_CONVERT_AUTH_PASSTHROUGH=1`, `CC_CONVERT_LOG_LEVEL`, `CC_CONVERT_LOG_FORMAT`.

Then hit it:

```bash
curl -X POST http://localhost:8787/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "claude-opus-4-7",
    "max_tokens": 100,
    "messages": [{"role":"user","content":"hello"}]
  }'
# → Anthropic-shape response, transparently sourced from the OpenAI upstream.
```

Streaming requests work the same — the SSE event stream you receive is
Anthropic-shape (`event: message_start`, `event: content_block_delta`, ...).

`GET /healthz` returns `ok` for liveness probes.
`GET /version` returns `{"name":"cc_convert","version":"..."}`.

### B) Pure Rust binary (no Python needed)

```bash
cargo build --release -p cc_convert_sidecar
export CC_CONVERT_UPSTREAM_URL="https://api.openai.com/v1/chat/completions"
export CC_CONVERT_UPSTREAM_API_KEY="sk-..."
./target/release/cc_convert_sidecar
```

Same env-var contract as the CLI. Use this when you want a single static
binary to drop on a machine that doesn't have Python.

## Two preset profiles

Both `ConvertOptions` (request) and `ResponseConvertOptions` / `StreamConvertOptions`
(response/stream) ship two presets you can choose between:

| Preset | What it does |
|---|---|
| `litellm_compat()` (default) | Byte-equivalent to LiteLLM's `AnthropicAdapter`. Use this when you're replacing LiteLLM in an existing pipeline and want zero behavioural drift. |
| `pragmatic()` / `anthropic_native()` | Closer to the published Anthropic SSE spec and what most OpenAI-compatible servers actually expect: `stream_options: {include_usage: true}` injected, `max_completion_tokens` used for o1/o3/o4/gpt-5, `stop` instead of `stop_sequences`, eager content_block opening, `ping` events, etc. |

## Test coverage (51 Rust tests + 22 Python tests, all passing)

```
crates/cc_convert_core/
├── src/                        ← 3 unit tests (tool name truncation)
└── tests/
    ├── request_translation.rs  ← 20 unit tests (cases 1–20)
    ├── response_translation.rs ← 7 unit tests (cases 21–26 + tool name round-trip)
    ├── stream_translation.rs   ← 5 unit tests (cases 27–31)
    ├── parity_litellm.rs       ← 2 LiteLLM request parity tests (32 fixtures)
    ├── parity_response.rs      ← 1 LiteLLM response parity test (6 fixtures)
    ├── parity_stream.rs        ← 1 LiteLLM stream parity test (2 fixtures, 3 documented quirks excluded)
    └── vendor_quirks.rs        ← 12 vLLM + SGLang quirk tests
crates/cc_convert_sidecar/tests/
└── integration.rs              ← 3 HTTP integration tests (non-streaming, streaming, 4xx)
python/tests/
└── test_parity.py              ← 22 pytest tests (LiteLLM parity through the wheel + smoke)
```

## Translation rules (high level)

Source of truth: LiteLLM's `AnthropicAdapter.translate_anthropic_to_openai`.

| Anthropic | → | OpenAI |
|---|---|---|
| `system: string` | | leading `{role:"system", content:<string>}` |
| `system: [{type:"text", text, cache_control?}]` | | leading `{role:"system", content:[{type:"text", text}, ...]}` (cache_control dropped) |
| user `text` block | | `{type:"text", text}` |
| user `image` (base64) | | `{type:"image_url", image_url:{url:"data:<media>;base64,<data>"}}` |
| user `image` (url) | | `{type:"image_url", image_url:{url}}` |
| user `tool_result` | | separate `{role:"tool", tool_call_id:<id>, content}` message, BEFORE any user text in that message; one tool message per tool_use_id |
| assistant `tool_use` | | entry in `tool_calls: [{id, type:"function", function:{name, arguments: JSON.stringify(input)}}]` |
| assistant `thinking` | | `thinking_blocks: [...]` on the assistant message (LiteLLM behaviour, opt out with `preserve_thinking_blocks=false`) |
| `max_tokens` | | `max_tokens` (LiteLLM-compat); opt in to `max_completion_tokens` for `o1*/o3*/o4*/gpt-5*` |
| `stop_sequences` | | `stop_sequences` (passthrough, LiteLLM-compat); opt in to `stop` |
| `top_k` | | passed through (LiteLLM behaviour; opt out via `drop_top_k`) |
| `tools` | | `[{type:"function", function:{name, description, parameters: input_schema}}]`; names >64 chars truncated to `{55-prefix}_{8-hex-sha}` |
| `tool_choice:{type:"any"}` | | `"required"` |
| `tool_choice:{type:"tool", name}` | | `{type:"function", function:{name}}` |
| `metadata.user_id` | | `user` |
| `thinking.budget_tokens` | | `reasoning_effort` (≥10000→high, ≥5000→medium, ≥2000→low, else minimal) |
| `cache_control` (any block) | | dropped |

Response side (OpenAI → Anthropic):

| OpenAI | → | Anthropic |
|---|---|---|
| `id` | | passed through (LiteLLM); opt into `chatcmpl-→msg_` rewrite |
| `choices[0].message.content` | | `{type:"text", text}` block |
| `choices[0].message.tool_calls` | | `{type:"tool_use", id, name, input: JSON.parse(arguments)}` blocks; tool name restored via map |
| `choices[0].message.reasoning_content` / `reasoning` | | `{type:"thinking", thinking}` block (accepts both — vLLM uses `reasoning`) |
| `finish_reason: stop\|length\|tool_calls\|content_filter\|abort` | | `stop_reason: end_turn\|max_tokens\|tool_use\|end_turn\|end_turn` |
| `usage.prompt_tokens` / `completion_tokens` | | `usage.input_tokens` / `output_tokens` (LiteLLM subtracts cached) |
| `usage.prompt_tokens_details.cached_tokens` | | `usage.cache_read_input_tokens` |

Streaming side: an OpenAI SSE chunk stream becomes a sequence of
`message_start` → `[ping]` → `content_block_start` → ... → `content_block_stop`
→ `message_delta` → `message_stop` events, with `text_delta` /
`input_json_delta` / `thinking_delta` for text / tool_call / reasoning
content respectively. Parallel tool_calls get distinct content-block
indices. Streams that end without a `finish_reason` are closed with
`stop_reason: end_turn`.

## Building from source

Requires Rust ≥ 1.75 and Python ≥ 3.8 (only for the wheel).

```bash
# If your environment needs an HTTP proxy for cargo/pip, set the usual env vars.
# (Optional — only needed in restricted networks.)
# export http_proxy=http://YOUR_PROXY:PORT
# export https_proxy=http://YOUR_PROXY:PORT
# export no_proxy="localhost,127.0.0.1"

# Rust core + sidecar
cargo build --release

# Rust tests (includes LiteLLM parity against committed goldens — no network)
cargo test --workspace

# Python wheel
cd python
pip install maturin
maturin build --release
pip install ../target/wheels/cc_convert-*.whl
pytest tests/
```

## Parity tests

Twenty plus twelve request fixtures live under `tests/fixtures/requests/` as
paired `anthropic_<name>.json` / `openai_<name>.json` files. Six response
and five stream fixtures sit under `responses/` and `streams/`. All golden
outputs were produced by running each input through LiteLLM and committed
to the repo so CI doesn't need network.

To regenerate goldens after a rule change:

```bash
pip install 'litellm>=1.0'
python scripts/seed_fixture_inputs.py        # initial 31 cases (only if missing)
python scripts/seed_extra_request_fixtures.py # 12 extra request cases
python scripts/regen_fixtures.py              # request goldens
python scripts/regen_response_fixtures.py     # response goldens
python scripts/regen_stream_fixtures.py       # stream goldens
cargo test --workspace                        # confirm parity still holds
```

## Layout

```
crates/
  cc_convert_core/      Pure-Rust translation library, no I/O.
  cc_convert_py/        PyO3 bindings → Python wheel (cc_convert._native).
  cc_convert_sidecar/   axum HTTP proxy binary + integration tests.
python/
  cc_convert/           Python package (re-exports the native module).
  tests/                pytest parity tests against the wheel.
scripts/
  seed_fixture_inputs.py        Generate INPUT fixtures (cases 1–31).
  seed_extra_request_fixtures.py Generate extra INPUT fixtures (cases 32–43).
  regen_fixtures.py             Run LiteLLM to produce request goldens.
  regen_response_fixtures.py    Same for responses.
  regen_stream_fixtures.py      Same for streams.
tests/
  fixtures/             Committed golden parity fixtures.
```

## Known gaps / non-goals (v1)

- **Hosted Anthropic tools** (`web_search`, `computer`, `bash`,
  `text_editor`) are not translated to OpenAI equivalents — they are
  passed through. v1.1 may map `web_search` to OpenAI's
  `web_search_options`.
- **`stop_sequence` detection** on responses is not implemented. None of
  the reference libs do it either; OpenAI doesn't surface the matched
  stop string. vLLM does via `stop_reason` (matched string) and SGLang via
  `matched_stop` — translating these into Anthropic's `stop_sequence`
  field would be straightforward to add but is not in v1.
- **Documented LiteLLM stream quirks**: three of our five stream fixture
  goldens diverge from spec because LiteLLM's `AnthropicStreamWrapper`
  produces non-spec output (merging parallel tool_calls into one block,
  silently truncating streams with no finish_reason, conflating
  reasoning+text into one block). We follow the spec; see
  `tests/parity_stream.rs:LITELLM_QUIRKS_TO_SKIP` for details.
- **Anthropic `[DONE]` sentinel**: real Anthropic SSE does *not* emit
  `data: [DONE]\n\n`. We don't either. We do *accept* it on input from
  upstream OpenAI as the end-of-stream marker.

## License

MIT OR Apache-2.0.
