# cc_convert

Anthropic Messages API ↔ OpenAI Chat Completions protocol converter.
Rust core via PyO3 + a Python CLI sidecar.

```bash
pip install cc-convert
```

```python
import cc_convert
openai_req, tool_map = cc_convert.translate_request(anthropic_request_dict)
# POST openai_req to any OAI-compatible /v1/chat/completions endpoint
anthropic_resp = cc_convert.translate_response(
    openai_response_dict,
    original_model="claude-opus-4-7",
    tool_name_map=tool_map,
)
```

Or use the CLI as a transparent sidecar proxy:

```bash
cc_convert serve --listen 0.0.0.0:8787 --upstream-url http://your-oai-host:8000
```

Then point any Anthropic-API client (Claude Code, claude-py, etc.) at
`http://localhost:8787`.

See full docs and examples at
[github.com/yitianlian/cc_convert](https://github.com/yitianlian/cc_convert)
([中文文档](https://github.com/yitianlian/cc_convert/blob/main/USAGE.zh-CN.md)).

## What it does

- **Anthropic request → OpenAI request** (single text collapsed to string,
  tools, tool_choice, system, multipart content, thinking → reasoning_effort,
  cache_control / hosted tools / server_tool_use blocks dropped or
  translated as appropriate).
- **OpenAI response → Anthropic response** (id rewrite, content blocks,
  tool_calls → tool_use, reasoning_content / reasoning → thinking block,
  finish_reason mapping, usage with cached_tokens accounting).
- **OpenAI SSE → Anthropic SSE** (full event sequence: message_start →
  content_block_start → deltas → content_block_stop → message_delta →
  message_stop).
- Compatible with real SGLang / vLLM / DeepSeek upstreams; handles their
  quirks (`reasoning_content: null` on every chunk, `id: null` on
  continuation tool_call chunks, `matched_stop`, `metadata.weight_version`,
  etc.) without breaking.

## License

MIT OR Apache-2.0.
