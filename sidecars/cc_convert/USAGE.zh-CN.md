# 使用文档

cc_convert 三种使用方式 + 测试与开发流程。

> 英文文档见 [README.md](README.md)。

## 一、当作 Python 库用

```bash
pip install cc_convert
```

```python
import cc_convert

# 把 Anthropic 格式请求转成 OpenAI 格式
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
# openai_req 就可以发给任何 OAI 兼容的 /v1/chat/completions

# 上游响应回来后,转回 Anthropic 格式
openai_resp = {...}   # 上游返回的
anthropic_resp = cc_convert.translate_response(
    openai_resp,
    original_model="claude-opus-4-7",
    tool_name_map=tool_map
)

# 流式版本
translator = cc_convert.StreamTranslator("claude-opus-4-7", tool_map)
for openai_chunk in upstream_sse_stream:        # 每个是 dict
    for anthropic_event in translator.push(openai_chunk):
        # anthropic_event 是 dict,类型有:
        # message_start / ping / content_block_start / content_block_delta /
        # content_block_stop / message_delta / message_stop
        emit_to_client(anthropic_event)
for trailing in translator.finish():
    emit_to_client(trailing)
```

### 两种翻译档位

```python
# Pragmatic(默认):贴合真实 OAI 上游(vLLM/SGLang strict mode)
#  - 单 text 内容折叠成字符串(很多上游不收 list content)
#  - reasoning_effort 自动从 thinking.budget_tokens 桶化
#  - max_completion_tokens for o1/o3/o4/gpt-5
#  - stream_options.include_usage 自动注入
cc_convert.translate_request(req)
cc_convert.translate_request(req, mode="pragmatic")

# LiteLLM-compat:byte-equivalent 平替 LiteLLM AnthropicAdapter
cc_convert.translate_request(req, mode="litellm_compat")
```

## 二、当作命令行 sidecar(HTTP 反向代理)用

```bash
# 装好 wheel 之后,有 cc_convert 命令
pip install cc_convert

# Proxy 模式:接 Anthropic 请求,转发到 OAI 后端,翻回 Anthropic
cc_convert serve \
    --listen 0.0.0.0:8787 \
    --upstream-url http://YOUR_UPSTREAM_HOST:8000 \
    --upstream-key sk-xxx                       # 可选,不传也行(本地上游)

# 然后 Claude Code / claude-py / cline 等客户端指过来:
export ANTHROPIC_BASE_URL=http://localhost:8787
claude   # 它现在以为后端是 Anthropic,实际是你的 OAI 上游
```

### CLI 常用参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--mode {proxy,rpc}` | `proxy` | proxy 转发,rpc 只翻译不发 |
| `--listen HOST:PORT` | `0.0.0.0:8787` | 监听地址 |
| `--upstream-url URL` | `$CC_CONVERT_UPSTREAM_URL` | 后端 URL(自动补 `/v1/chat/completions`) |
| `--upstream-key KEY` | `$CC_CONVERT_UPSTREAM_API_KEY` | Bearer token |
| `--auth-passthrough` | off | 转发客户端 Authorization 而不用 `--upstream-key` |
| `--compat-mode {pragmatic,litellm_compat}` | `pragmatic` | 翻译档位(同 Python lib) |
| `--log-level {debug,info,warning,error}` | `info` | 日志级别 |
| `--log-format {text,json}` | `text` | json 是一行一对象,方便日志采集 |
| `-v` / `-vv` | | 等价 `--log-level info/debug` |
| `--quiet` | off | 不打访问日志 |
| `--version` | | 打印版本 |

支持的端点路径:`/v1/messages`、`/messages`、`/anthropic/v1/messages`(任何以 `/messages` 或 `/v1/messages` 结尾的路径都识别)。

### One-shot 命令行翻译(不起 server)

```bash
# Anthropic 请求 → OAI 请求
cat anthropic_req.json | cc_convert translate --direction cc-to-oai

# OAI 响应 → Anthropic 响应
cat openai_resp.json | cc_convert translate \
    --direction oai-to-cc \
    --original-model claude-opus-4-7
```

### RPC 模式(只翻译,不转发)

```bash
cc_convert serve --mode rpc --listen 127.0.0.1:8788

# 然后可以打 HTTP 请求做翻译
curl -X POST http://127.0.0.1:8788/translate/cc-to-oai -d '{...anthropic request...}'
curl -X POST http://127.0.0.1:8788/translate/oai-to-cc -d '{"openai_response":{...}, "original_model":"...", "tool_map":{}}'
```

## 三、当作纯 Rust 库 / 静态二进制

```bash
# 纯 Rust 二进制(不依赖 Python)
cargo build --release -p cc_convert_sidecar

export CC_CONVERT_UPSTREAM_URL="http://YOUR_UPSTREAM_HOST:8000/v1/chat/completions"
export CC_CONVERT_UPSTREAM_API_KEY="sk-..."
./target/release/cc_convert_sidecar
```

环境变量和 Python CLI 一致(都用 `CC_CONVERT_*` 前缀)。

---

## 上游需要的配置

cc_convert 不做"客户端兜底解析"。如果上游模型把 `<think>` / `<tool_call>` 留在 content 里没拆,我们就如实透传。**真正的修复在上游**:

### SGLang 启动参数

| 你看到什么现象 | 上游加什么 flag |
|---|---|
| `<think>...</think>` 留在 content,`reasoning_content: null` | `--reasoning-parser qwen3`(或 `deepseek-r1`、`hunyuan` 等 — 见下表) |
| `<tool_call>...</tool_call>` 留在 content,`tool_calls: null` | `--tool-call-parser qwen25`(或 `hermes`、`pythonic` 等 — 见下表) |

#### reasoning parser 选项

`deepseek-r1` `deepseek-v3` `deepseek-v4` `qwen3` `qwen3-thinking` `glm45` `hunyuan` `gpt-oss` `kimi` `kimi_k2` `mistral` `mimo` `poolside_v1` `minimax` `minimax-append-think` `step3` `step3p5` `interns1` `nemotron_3` `gemma4`

#### tool-call parser 选项

`qwen25` `qwen` `qwen3_coder` `hermes` `deepseekv3` `deepseekv31` `deepseekv32` `deepseekv4` `llama3` `mistral` `kimi_k2` `glm` `glm45` `glm47` `pythonic` `gpt-oss` `cohere_command4` `lfm2` `minicpm5` `mimo` `step3` `step3p5` `minimax-m2` `trinity` `interns1` `hunyuan` `gigachat3` `gemma4`

启动示例:

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen3-model \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen25 \
    ...
```

---

## 测试与开发流程

### 跑 Rust 测试

```bash
cargo test --workspace        # 全跑(63 core + 3 sidecar)
cargo test -p cc_convert_core --tests          # 只 core
cargo test -p cc_convert_sidecar --test integration   # 只 sidecar
```

### 跑 Python 测试

```bash
cd python
maturin build --release       # 出 wheel 到 ../target/wheels/
pip install --force-reinstall ../target/wheels/cc_convert-*.whl
pytest tests/                 # 69 个测试
```

### 编辑代码后快速重测

```bash
cargo check -p cc_convert_core      # 5 秒内类型检查,日常开发用这个
cargo test -p cc_convert_core --tests --lib   # 30 秒,跑单元 + 翻译规则
```

### 跑线上 round-trip

playground 是一个开箱即用的端到端测试场:

```bash
# 准备好上游(改 URL/model 即可)
python playground/run_roundtrip.py \
    --upstream http://YOUR_UPSTREAM_HOST:8000 \
    --model /model

# 数据在 playground/runs/<timestamp>/<fixture_name>/ 下,
# 每个 fixture 一组 4 个 json:
#   1_anthropic_request.json    源 Anthropic 请求(verbatim)
#   2_oai_request.json          cc_convert 翻译后发给上游的
#   3_oai_response.json         上游返回的原始 OAI 响应
#   4_anthropic_response.json   cc_convert 翻译回 Anthropic 的最终响应
#
# 加上 meta.json(状态、延迟、HTTP code)和 _summary.json(汇总)
```

只跑一个 fixture:

```bash
python playground/run_roundtrip.py --upstream http://... --model /model --only agent_loop
```

### 7 个 fixture 各测什么

| Fixture | 测什么 |
|---|---|
| `02_reasoning_request` | extended thinking,`thinking.budget_tokens` → `reasoning_effort` |
| `03_forced_tool` | `tool_choice:any` → OpenAI `required` |
| `05_simple_text` | 单轮纯文本,baseline |
| `07_multi_turn_text` | 5 轮纯文本对话历史 |
| `08_agent_loop_with_tools` | **5 轮 agent loop**:assistant 调 2 个 tool → user 给 2 个 tool_result → 模型继续 |
| `09_long_response` | 4000 tokens 长生成,测大输出和长延迟 |
| `10_parallel_tools_text_only` | 一次请求触发多个 parallel tool_use |

### Git push 前的安全审计

可选,但建议在 push 之前过一遍敏感信息:

```bash
# 简单 grep 内网代理 / 路径 / API key
grep -rIEn "httpproxy|/workspace/|/root/|sk-[a-zA-Z0-9]{10,}|172\.27|10\.180" \
    --exclude-dir=target --exclude-dir=__pycache__ --exclude-dir=.git \
    --exclude-dir=playground/runs \
    --include="*.rs" --include="*.py" --include="*.toml" --include="*.md" .
```

playground/runs/ 已经在 .gitignore 里,任何线上测试结果不会进 git。

---

## 常见问题

**Q: 我看上去的输出 `reasoning_content` 是 null,但 content 里有 `<think>` 怎么办?**
A: 上游没启 `--reasoning-parser`,让运维加。cc_convert 不做客户端兜底。

**Q: tool_calls 也是 null,但 content 里有 `<tool_call>` 标签?**
A: 同上,上游没启 `--tool-call-parser`,加 `qwen25` 或 `hermes`。

**Q: 上游 503 / connection refused,是 cc_convert 的问题吗?**
A: 不是。看 `playground/runs/<ts>/<fx>/error.txt`,如果说 "No available workers" 或 "Connection refused",是上游 worker 不稳定 — 我们的请求已经合规发出去了。

**Q: Claude Code / OpenCode 发的额外字段(`output_config` / `speed` / `container` 等)会被丢吗?**
A: 不会。从 commit `0be4d80` 起,所有未知字段都保留在 `AnthropicRequest.extra` 里。其中 `output_config.effort` 和 `service_tier` 会真正翻译到 OpenAI 对应字段;其他暂时保留供未来 Anthropic-target proxy 用。

**Q: 默认行为是不是 LiteLLM 兼容?**
A: 不是。默认是 `pragmatic`,贴合真实 OAI 上游。需要 LiteLLM byte-parity 时用 `mode="litellm_compat"` 或 `--compat-mode litellm_compat`。
