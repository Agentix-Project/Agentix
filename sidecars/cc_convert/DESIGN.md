# cc_convert design principle: source of truth for each translation direction

## Direction A: Anthropic → OpenAI (request translation)

**Source of truth**: what real OpenAI-compatible model servers actually accept on the wire, in this priority:

1. **vLLM** — `vllm/entrypoints/openai/chat_completion/protocol.py` and `chat_utils.py`
   on GitHub `vllm-project/vllm`. Authoritative for self-hosted production.
2. **SGLang** — `python/sglang/srt/entrypoints/openai/protocol.py` and
   `serving_chat.py` on `sgl-project/sglang`. Used by 启智 / a lot of CN
   teams.
3. **DeepSeek hosted API** — `api-docs.deepseek.com`. The strictest in the
   wild (rejects `reasoning_content` on input with HTTP 400).
4. **OpenAI Python SDK request types** — `openai/openai-python` repo,
   `src/openai/types/chat/chat_completion_*.py`. The official schema.
5. **Popular chat templates** on Hugging Face — `tokenizer_config.json` for
   DeepSeek-R1, Qwen3, QwQ, Llama 3.3. These tell us which message fields
   the model ACTUALLY consumes once rendered.

**NEVER** use LiteLLM's intermediate adapter output as the source of truth.
LiteLLM has provider-side transformations that strip/rewrite fields between
its `AnthropicAdapter` and the wire. We previously made this mistake by
forwarding LiteLLM's `thinking_blocks` shape on the wire — no real upstream
consumes it.

LiteLLM parity is supported as an **opt-in compatibility mode**
(`litellm_compat`) for drop-in replacement, but the **default** must match
what real upstreams accept.

## Direction B: OpenAI → Anthropic (response translation)

**Source of truth**: the Anthropic Messages API official documentation.

1. **Anthropic docs** — `docs.anthropic.com/en/api/messages` (request +
   response shapes), `docs.anthropic.com/en/api/messages-streaming` (SSE
   event shapes). Authoritative.
2. **Anthropic Python SDK types** — `anthropics/anthropic-sdk-python` repo,
   `src/anthropic/types/message.py` etc.
3. **Anthropic client tooling** (Claude Code, claude-py) — what they
   actually parse. If a field is in the docs but no SDK reads it, we don't
   need to emit it.

Where Anthropic adds new features (extended thinking, hosted tools,
prompt caching), follow the Anthropic spec verbatim — do NOT inherit
LiteLLM's interpretation.

## Outstanding audit items

Anywhere the current Rust translator was shaped against LiteLLM intermediate
output needs to be re-audited against real upstreams / Anthropic docs:

- [x] **`thinking_blocks` field** — was LiteLLM-internal, no real consumer.
      Fixed: emit `reasoning_content: string` by default (vLLM/SGLang/Qwen3
      consume), `LiteLLMThinkingBlocks` and `Drop` as opt-in modes.
- [ ] **`cache_control` propagation** — currently dropped. Verify
      Anthropic-via-OpenAI proxies (when target is itself Anthropic) want
      it preserved.
- [ ] **`tool_choice` field on streaming** — confirm vLLM/SGLang `required`
      vs `any` semantics.
- [ ] **`top_k`** — currently passed through (LiteLLM behaviour). vLLM
      accepts it in `SamplingParams`; OpenAI spec rejects it. Make default
      drop for OpenAI targets, pass through for vLLM/SGLang via opt-in.
- [ ] **`stop_sequences` vs `stop`** — both wire names; vLLM accepts
      `stop`, SGLang accepts both. Verify.
- [ ] **`max_tokens` vs `max_completion_tokens`** — only `o1*/o3*/o4*/gpt-5*`
      strictly require `max_completion_tokens`. vLLM/SGLang accept both for
      any model. Verify.
- [ ] **`stream_options.include_usage`** — confirm SGLang and DeepSeek
      both honour this; some self-hosted servers ignore it.
- [ ] **`metadata`** — Anthropic-only `metadata.user_id` → OpenAI `user`.
      Confirmed.
- [ ] **`thinking.budget_tokens`** → `reasoning_effort` bucketing — only
      applies to OpenAI o-series and a few others. vLLM may want a raw
      `extra_body.reasoning.budget_tokens`. Audit.
- [ ] **Response side: `reasoning_content` vs `reasoning`** — already
      aliased in the deserializer. Good.
- [ ] **Response side: `stop_sequence` field** — should be the matched
      stop string when known. vLLM has `stop_reason`, SGLang has
      `matched_stop` — currently ignored. Anthropic clients sometimes
      check this.
- [ ] **Streaming SSE event shapes** — re-verify against Anthropic
      official streaming docs (not just LiteLLM's `AnthropicStreamWrapper`,
      which has documented bugs around parallel tool_calls etc.)

## Process going forward

When changing any translation rule:
1. Check the **real-upstream** source (vLLM/SGLang/Anthropic docs) first.
2. Decide what the default should be based on what 80% of real upstreams
   accept.
3. If LiteLLM disagrees with reality, add a `--compat-mode litellm_compat`
   opt-in for the LiteLLM behaviour. The default follows reality.
