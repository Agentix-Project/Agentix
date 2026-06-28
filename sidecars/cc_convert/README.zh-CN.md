# cc_convert

> English docs: [README.md](README.md)
>
> 详细用法 / Usage details: [USAGE.zh-CN.md](USAGE.zh-CN.md) ([English](USAGE.md))

**Anthropic 与 OpenAI Chat Completions 协议的双向转换器**,用 Rust 写的核心。
让客户端继续按 Anthropic Messages API 的样子调(`system` + content blocks +
`tool_use` / `tool_result` + Anthropic SSE),实际后端是 OpenAI 兼容的服务器
(也可反向)。

已对照三个参考实现验证:
- [LiteLLM] 的 `AnthropicAdapter`(主要的 parity oracle,覆盖 request、response、stream)
- [1rgs/claude-code-proxy]、[maxnowack/anthropic-proxy](交叉对照)
- [THUDM/slime] 的 anthropic adapter(中文 RL 生态)

并针对自建服务器的**非标准行为**做了硬化:
- **vLLM**:用 `reasoning`(不是 `reasoning_content`);多出 `stop_reason`、
  `prompt_logprobs`、`kv_transfer_params` 等字段;第一个 stream chunk 只有 role。
- **SGLang**:在 tool_call 续传 chunk 上发送 `id: null` 和 `function.name: null`;
  每一帧都带 `reasoning_content: null`;额外的 `matched_stop`、`metadata`、`sglext`;
  `finish_reason: "abort"`;kimi_k2 工具 ID 用 `functions.<name>:<idx>` 形式。

[LiteLLM]: https://github.com/BerriAI/litellm
[1rgs/claude-code-proxy]: https://github.com/1rgs/claude-code-proxy
[maxnowack/anthropic-proxy]: https://github.com/maxnowack/anthropic-proxy
[THUDM/slime]: https://github.com/THUDM/slime/tree/main/slime/agent/adapters

## 两种部署方式

| 模式 | 说明 | 何时用 |
|---|---|---|
| Python 包 | `pip install cc_convert`,`import cc_convert` | Python 应用里直接调用,dict 进 dict 出 |
| HTTP sidecar | `cc_convert_sidecar` 可执行文件,监听 `/v1/messages`(Anthropic 格式),反向代理到上游 OpenAI 兼容服务器 | 把 Anthropic API 客户端(Claude Code、claude-py 等)接到非 Anthropic 后端 |

两种方式共享同一份 Rust 翻译核心(`cc_convert_core`),行为完全一致。

## Python 用法

```python
import cc_convert

# 1) Anthropic 请求 → OpenAI 请求
anthropic_req = {
    "model": "gpt-4o-mini",
    "max_tokens": 500,
    "system": "回答简洁。",
    "tools": [{
        "name": "get_weather",
        "description": "查询城市天气",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }],
    "tool_choice": {"type": "any"},
    "messages": [{"role": "user", "content": "东京天气如何?"}],
}
openai_req, tool_map = cc_convert.translate_request(anthropic_req)
# → 把 openai_req POST 到任意 OpenAI 兼容的 /v1/chat/completions

# 2) OpenAI 响应 → Anthropic 响应
openai_resp = {...}   # 上游返回的
anthropic_resp = cc_convert.translate_response(
    openai_resp, original_model="claude-opus-4-7", tool_name_map=tool_map
)

# 3) 流式:OpenAI SSE chunks → Anthropic SSE 事件
translator = cc_convert.StreamTranslator("claude-opus-4-7", tool_map)
for openai_chunk in upstream_sse_chunks:           # 每个都是 dict
    for anthropic_event in translator.push(openai_chunk):
        emit_to_client(anthropic_event)            # type: message_start / content_block_* / message_delta / message_stop
for trailing in translator.finish():
    emit_to_client(trailing)
```

`translate_request` 返回 `(openai_request, tool_name_map)`。第一个就是要 POST 给
上游的请求体;第二个是 `dict[str, str]`,把"截断后的工具名 → 原始工具名"映射保存
下来,响应端用它还原超过 OpenAI 64 字符上限被截断的工具名。

## Sidecar 用法

两种跑法,任选其一:

### A) 内置 CLI(Python wheel 自带)——最简单

`pip install cc_convert` 之后,直接有一个 `cc_convert` 命令:

```bash
# proxy 模式:监听 Anthropic 请求,转发到 OpenAI 上游,翻回 Anthropic 响应
cc_convert serve \
    --listen 0.0.0.0:8787 \
    --upstream-url https://api.openai.com/v1/chat/completions \
    --upstream-key sk-...

# 指向 vLLM / SGLang / DeepSeek 之类自建服务
cc_convert serve --upstream-url http://localhost:8000/v1/chat/completions -v

# 纯翻译 RPC 模式(不转发,只翻译)
cc_convert serve --mode rpc --listen 127.0.0.1:8788

# 一次性 JSON 进 JSON 出(不起 server)
cat anthropic_req.json | cc_convert translate --direction cc-to-oai
cat openai_resp.json  | cc_convert translate --direction oai-to-cc --original-model claude-opus-4-7
```

常用参数:

| 参数 | 默认 | 含义 |
|---|---|---|
| `--mode {proxy,rpc}` | `proxy` | `proxy`:在 `--cc-path` 上接 Anthropic 请求,转发到 `--upstream-url`,翻回 Anthropic 返回。`rpc`:无状态 `/translate/cc-to-oai` 和 `/translate/oai-to-cc` 两个端点。 |
| `--listen HOST:PORT` | `0.0.0.0:8787` | 监听地址 |
| `--upstream-url URL` | `$CC_CONVERT_UPSTREAM_URL` | (proxy)OpenAI 兼容的 `/v1/chat/completions` URL |
| `--upstream-key KEY` | `$CC_CONVERT_UPSTREAM_API_KEY` | (proxy)Bearer token,发给上游 |
| `--auth-passthrough` | off | 改成透传客户端的 `Authorization` / `x-api-key`,不用 `--upstream-key` |
| `--cc-path PATH` | `/v1/messages` | (proxy)接 Anthropic 请求的路径 |
| `--cc-to-oai-path PATH` | `/translate/cc-to-oai` | (rpc)请求端翻译路径 |
| `--oai-to-cc-path PATH` | `/translate/oai-to-cc` | (rpc)响应端翻译路径 |
| `--log-level LEVEL` | `info` | `debug` / `info` / `warning` / `error` |
| `--log-format FORMAT` | `text` | `text`(给人看)或 `json`(一行一个 JSON 对象,方便采集) |
| `-v` / `-vv` | | 快捷写法,等价于 `--log-level info` / `debug` |
| `--quiet` | off | 不打访问日志(只剩错误日志) |
| `--version` | | 打印版本退出 |

所有参数都有对应的环境变量默认值:`CC_CONVERT_MODE`、`CC_CONVERT_LISTEN_ADDR`、
`CC_CONVERT_UPSTREAM_URL`、`CC_CONVERT_UPSTREAM_API_KEY`、
`CC_CONVERT_AUTH_PASSTHROUGH=1`、`CC_CONVERT_LOG_LEVEL`、`CC_CONVERT_LOG_FORMAT`。

调起来:

```bash
curl -X POST http://localhost:8787/v1/messages \
  -H 'content-type: application/json' \
  -d '{
    "model": "claude-opus-4-7",
    "max_tokens": 100,
    "messages": [{"role":"user","content":"你好"}]
  }'
# → 返回 Anthropic 格式响应,实际由 OpenAI 上游产生
```

流式请求一样支持(SSE 事件流是 Anthropic 格式)。
`GET /healthz` 返回 `ok` 给 liveness probe;`GET /version` 返回版本信息。

### B) 纯 Rust 二进制(不依赖 Python)

```bash
cargo build --release -p cc_convert_sidecar
export CC_CONVERT_UPSTREAM_URL="https://api.openai.com/v1/chat/completions"
export CC_CONVERT_UPSTREAM_API_KEY="sk-..."
./target/release/cc_convert_sidecar
```

参数和环境变量与 CLI 完全一致。需要在一台没有 Python 的机器上跑一个静态二进制时用这个。

## 两个预设档位

`ConvertOptions`(请求)、`ResponseConvertOptions` / `StreamConvertOptions`
(响应/流)都提供两个预设:

| 预设 | 行为 |
|---|---|
| `litellm_compat()`(默认) | 与 LiteLLM 的 `AnthropicAdapter` 字节级等价(忽略 null 字段差异)。在已有 LiteLLM pipeline 里平替时用这个,行为零漂移。 |
| `pragmatic()` / `anthropic_native()` | 更贴近 Anthropic SSE 官方规范、以及大多数 OpenAI 兼容服务器实际期望的形态:`stream` 时注入 `stream_options: {include_usage: true}`、o1/o3/o4/gpt-5 用 `max_completion_tokens`、stop 字段输出 `stop`、content_block 急切打开、有 `ping` 事件等。 |

## 测试覆盖(51 个 Rust + 22 个 Python,全过)

```
crates/cc_convert_core/
├── src/                        ← 3 个单元测试(工具名截断)
└── tests/
    ├── request_translation.rs  ← 20 个单元测试(case 1–20)
    ├── response_translation.rs ← 7 个单元测试(case 21–26 + 工具名往返)
    ├── stream_translation.rs   ← 5 个单元测试(case 27–31)
    ├── parity_litellm.rs       ← 2 个 LiteLLM 请求 parity 测试(32 个 fixture)
    ├── parity_response.rs      ← 1 个 LiteLLM 响应 parity 测试(6 个 fixture)
    ├── parity_stream.rs        ← 1 个 LiteLLM 流 parity 测试(2 个 fixture,3 个文档化的 LiteLLM 缺陷排除)
    └── vendor_quirks.rs        ← 12 个 vLLM + SGLang 的 quirk 测试
crates/cc_convert_sidecar/tests/
└── integration.rs              ← 3 个 HTTP 集成测试(非流式、流式、4xx)
python/tests/
└── test_parity.py              ← 22 个 pytest(走 wheel 跑 LiteLLM parity + 烟雾测试)
```

## 翻译规则(高层)

真理来源:LiteLLM 的 `AnthropicAdapter.translate_anthropic_to_openai`。

| Anthropic | → | OpenAI |
|---|---|---|
| `system: string` | | 顶部加 `{role:"system", content:<string>}` |
| `system: [{type:"text", text, cache_control?}]` | | 顶部加 `{role:"system", content:[{type:"text", text}, ...]}`(cache_control 丢弃) |
| user `text` 块 | | `{type:"text", text}` |
| user `image`(base64) | | `{type:"image_url", image_url:{url:"data:<media>;base64,<data>"}}` |
| user `image`(url) | | `{type:"image_url", image_url:{url}}` |
| user `tool_result` | | 单独的 `{role:"tool", tool_call_id:<id>, content}` 消息,排在该 user 消息的任何 text 之前;每个 tool_use_id 对应一条 tool 消息 |
| assistant `tool_use` | | `tool_calls` 数组里加一项 `{id, type:"function", function:{name, arguments: JSON.stringify(input)}}` |
| assistant `thinking` | | 加到 assistant 消息的 `thinking_blocks` 数组(LiteLLM 行为,可用 `preserve_thinking_blocks=false` 关掉) |
| `max_tokens` | | `max_tokens`(LiteLLM-compat);可开启 reasoning 模型用 `max_completion_tokens` |
| `stop_sequences` | | `stop_sequences`(透传,LiteLLM-compat);可改成发 `stop` |
| `top_k` | | 透传(LiteLLM 行为,可 `drop_top_k` 关) |
| `tools` | | `[{type:"function", function:{name, description, parameters: input_schema}}]`;名字 >64 字符的截断成 `{55前缀}_{8位hex哈希}` |
| `tool_choice:{type:"any"}` | | `"required"` |
| `tool_choice:{type:"tool", name}` | | `{type:"function", function:{name}}` |
| `metadata.user_id` | | `user` |
| `thinking.budget_tokens` | | `reasoning_effort`(≥10000→high、≥5000→medium、≥2000→low、否则 minimal) |
| 任意块的 `cache_control` | | 丢弃 |

响应方向(OpenAI → Anthropic):

| OpenAI | → | Anthropic |
|---|---|---|
| `id` | | 透传(LiteLLM);可开启 `chatcmpl-→msg_` 重命名 |
| `choices[0].message.content` | | `{type:"text", text}` 块 |
| `choices[0].message.tool_calls` | | `{type:"tool_use", id, name, input: JSON.parse(arguments)}` 块;工具名走 map 还原 |
| `choices[0].message.reasoning_content` / `reasoning` | | `{type:"thinking", thinking}` 块(两个字段名都认,vLLM 用 `reasoning`) |
| `finish_reason: stop\|length\|tool_calls\|content_filter\|abort` | | `stop_reason: end_turn\|max_tokens\|tool_use\|end_turn\|end_turn` |
| `usage.prompt_tokens` / `completion_tokens` | | `usage.input_tokens` / `output_tokens`(LiteLLM 会减掉 cached 部分) |
| `usage.prompt_tokens_details.cached_tokens` | | `usage.cache_read_input_tokens` |

流式方向:OpenAI SSE chunk 流转成
`message_start` → `[ping]` → `content_block_start` → ... → `content_block_stop`
→ `message_delta` → `message_stop` 序列。文本对应 `text_delta`、工具调用对应
`input_json_delta`、reasoning 对应 `thinking_delta`。并行 tool_calls 会拿到不同
的 content_block index。流提前断掉(没有 finish_reason)时,我们自己补一个
`stop_reason: end_turn` 结尾。

## 从源码编译

需要 Rust ≥ 1.75,Python ≥ 3.8(只为打 wheel)。

```bash
# 如果你的环境 cargo/pip 需要走代理,自行设置;非受限网络可以忽略。
# export http_proxy=http://YOUR_PROXY:PORT
# export https_proxy=http://YOUR_PROXY:PORT
# export no_proxy="localhost,127.0.0.1"

# Rust 核心 + sidecar
cargo build --release

# Rust 测试(包含 LiteLLM parity 对照已提交的 golden,不需要联网)
cargo test --workspace

# Python wheel
cd python
pip install maturin
maturin build --release
pip install ../target/wheels/cc_convert-*.whl
pytest tests/
```

## Parity 测试

`tests/fixtures/requests/` 下放着 20 + 12 个请求 fixture,配对的
`anthropic_<name>.json` / `openai_<name>.json`。`responses/` 和 `streams/` 下
各有 6 个响应和 5 个流 fixture。所有 golden 都是先用 LiteLLM 跑出来并提交到
仓库的,所以 CI 不需要联网。

规则改了之后重新生成 golden:

```bash
pip install 'litellm>=1.0'
python scripts/seed_fixture_inputs.py        # 初始 31 个 case(只在缺失时跑)
python scripts/seed_extra_request_fixtures.py # 额外 12 个请求 case
python scripts/regen_fixtures.py              # 请求 golden
python scripts/regen_response_fixtures.py     # 响应 golden
python scripts/regen_stream_fixtures.py       # 流 golden
cargo test --workspace                        # 确认 parity 还在
```

## 目录结构

```
crates/
  cc_convert_core/      纯 Rust 翻译库,无 I/O
  cc_convert_py/        PyO3 binding → Python wheel(cc_convert._native)
  cc_convert_sidecar/   axum HTTP 反向代理二进制 + 集成测试
python/
  cc_convert/           Python 包(re-export 原生模块)
  tests/                pytest parity 测试,走 wheel
scripts/
  seed_fixture_inputs.py        生成 INPUT fixture(case 1–31)
  seed_extra_request_fixtures.py 生成额外 INPUT fixture(case 32–43)
  regen_fixtures.py             跑 LiteLLM 生成请求 golden
  regen_response_fixtures.py    同上,响应
  regen_stream_fixtures.py      同上,流
tests/
  fixtures/             提交进仓库的 golden parity fixture
```

## 已知缺口 / 非目标(v1)

- **Anthropic hosted tools**(`web_search`、`computer`、`bash`、`text_editor`)
  不转换成 OpenAI 等价物,直接透传。v1.1 可以考虑把 `web_search` 映射到
  OpenAI 的 `web_search_options`。
- **响应端 `stop_sequence` 检测** 没做。三个参考库都没做;OpenAI 也不会告诉
  你哪个 stop 命中了。但 vLLM 通过 `stop_reason`(命中的字符串)、SGLang 通过
  `matched_stop` 能给出来,要做成 Anthropic 的 `stop_sequence` 是平凡的,
  只是 v1 没加。
- **三个 LiteLLM 流式缺陷**:5 个流 fixture 里有 3 个的 golden 不符合 Anthropic
  spec(把并行 tool_calls 合并成一个块、流没 finish_reason 时静默截断、
  reasoning + text 合成一个块)。我们按 spec 实现,所以这 3 个 case 在
  `tests/parity_stream.rs:LITELLM_QUIRKS_TO_SKIP` 中跳过 parity,详见该常量
  附近的注释。
- **Anthropic `[DONE]` 哨兵**:真实 Anthropic SSE 不会发 `data: [DONE]\n\n`,
  我们也不发。但**输入**端会接收它,当作 OpenAI 流结束的标记。

## License

MIT OR Apache-2.0.
