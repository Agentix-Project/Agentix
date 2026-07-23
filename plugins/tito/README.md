# agentix-tito — TITO Gateway

An Agentix plugin (`import agentix.tito`) that records **token-aligned**
agent↔model trajectories. It sits between an agent and an OpenAI-compatible
inference backend (sglang or vLLM) as a session-scoped proxy, and accumulates
the exact prompt / completion token IDs of every turn — the trajectory an RL
trainer needs, with no host-side re-tokenization.

This is a **native implementation** of the TITO token-alignment engine
(`agentix.tito.engine`): no vendored training-framework code and no `sglang`
dependency. The engine tokenizes prompts itself with `transformers` +
`tokenizers` + a fixed Jinja chat template.

## TITO in one paragraph

TITO = *token-in, token-out*. Instead of re-tokenizing the rendered chat
transcript host-side (which can drift from what the model actually saw), the
gateway:

1. **pretokenizes** each request's messages to prompt token IDs and sends
   those to the backend (token-in),
2. reads the **exact completion token IDs** back from the backend's token
   dialect (token-out),
3. reuses the **byte-identical token prefix** across turns, tokenizing only the
   newly-appended non-assistant messages (tool/user/system) as a suffix in a
   synthetic context, and
4. on read, **audits** the accumulated trajectory against a from-scratch render
   (`compute_session_mismatch`) so any tokenizer drift is detected, not hidden.

The algorithm is **model-agnostic** (base `TITOTokenizer`); a model family is a
fixed chat template plus a tiny boundary fixup — e.g. `Qwen3TITOTokenizer`
re-inserts the `\n` after `<|im_end|>` that the model omits when it stops.

## Backend kinds

The token dialect is selected by `--backend-kind` (`TITOGatewayConfig.backend_kind`):

- **`sglang`** (default) — one chat-completions call per turn; the prompt rides
  sglang's `input_ids` extension and the completion IDs come back in
  `meta_info.output_token_logprobs`.
- **`vllm`** — requires vLLM ≥ 0.24.0 (the first release with the derender
  endpoints). vLLM's chat-completions endpoint **silently ignores** an
  `input_ids` field, so each turn is the token-native three-call chain:
  `/v1/chat/completions/render` (chat request → `GenerateRequest`, resolving
  sampling params server-side; its from-scratch token IDs are discarded) →
  `/inference/v1/generate` on the session's exact prompt IDs →
  `/v1/chat/completions/derender` (raw token output → chat completion via the
  server's tool/reasoning parsers). Launch the vLLM server with the parser
  flags that match your model (e.g. `--enable-auto-tool-choice
  --tool-call-parser hermes --reasoning-parser qwen3`), or tool calls come
  back as plain text content.
  The request body must include `model` (derender requires it).

## Install

It is a member of the Agentix uv workspace, installed editable with the rest:

```bash
uv sync --all-packages --all-extras
```

Its runtime deps (`transformers`, `tokenizers`, `jinja2`, …) are isolated to
this plugin — agentix core and other plugins never pull them.

## CLI

```bash
agentix-tito serve \
  --hf-checkpoint Qwen/Qwen3-4B \
  --backend-url http://127.0.0.1:30000 \
  --tito-model qwen3 \
  --session-server-port 30001
```

`--tito-model` selects the tokenizer family (`qwen3`, or `default` for the
tokenizer's own template). `--backend-kind` selects the backend token dialect
(`sglang` default, or `vllm`). `--backend-url` may be omitted to auto-discover
a local backend (see `agentix.tito.discovery`). Run `agentix-tito serve -h`
for the full list.

## Per-turn record persistence — `tito.record.v1` (normative)

With `--record-dir DIR` (env `TITO_RECORD_DIR`), every committed turn appends
one `tito.record.v1` JSON line to `DIR/<session_id>.jsonl` and flushes it —
the file is complete up to the last committed turn even if the process dies
mid-rollout. **This section is the normative schema document: the gateway is
the producing side of the contract, and downstream consumers adapt to the
shape defined here.** The structure is flat (no nested `prompt`/`completion`
objects).

```jsonc
{
  "schema_version": "tito.record.v1",
  "session_id": "<gateway session id>",
  "thread_id": "<echo of the x-thread-id request header>",  // OPTIONAL: key omitted when the header is absent
  "request_id": "<echo of the x-request-id request header, or null>",
  "turn_index": 0,             // monotonic per session file; see gap semantics below
  "ts": 1750000000.0,          // unix seconds, request commit time
  "model": "<chat request model field, or null>",
  "backend_kind": "sglang" | "vllm",
  "sampling": { "temperature": 0.6, "top_p": 0.95, "max_tokens": 4096 },
  "tokenizer": {
    "checkpoint": "<hf checkpoint name/path the gateway loaded>",
    "tokenizer_sha256": "<64 hex>",
    "chat_template_sha256": "<64 hex>"
  },
  "prompt_token_ids": [ ... ],           // exactly what generation ran on
  "prompt_segments": [ {"start": 0, "end": 42, "source": "prefix"}, ... ],
  "completion_token_ids": [ ... ],       // exactly what was sampled
  "completion_logprobs": [ ... ],        // finite floats, len == len(completion_token_ids)
  "assistant_message": { ... },          // the parsed assistant turn as stored
  "finish_reason": "stop" | "tool_calls" | ... | null,
  "prefix_stable": true,
  "render_skew": null | {"equal": false, "first_divergence": 17}
}
```

Field semantics:

- `sampling` — the whitelisted sampling parameters lifted verbatim from the
  chat request body (`temperature`, `top_p`, `top_k`, `min_p`, `max_tokens`,
  `max_completion_tokens`, `frequency_penalty`, `presence_penalty`,
  `repetition_penalty`, `seed`, `stop`, `n`); keys absent from the request
  are absent here, so the empty object means "backend defaults".
- `tokenizer.tokenizer_sha256` — SHA-256 over the tokenizer **definition**
  bytes: for fast tokenizers the complete serialized `tokenizer.json`
  definition (`backend_tokenizer.to_str()` — vocab, merges, normalizer,
  added tokens); for slow tokenizers the sorted-JSON vocabulary. Two
  gateways report the same value iff they tokenize identically.
  `tokenizer.chat_template_sha256` hashes the chat template actually in
  effect (the fixed-template override when set, else the tokenizer's own).
- `prompt_segments.source` vocabulary (gateway-native, exhaustive):
  `render` (a from-scratch chat-template render — the first turn, or the
  fallback when the prefix window was outrun), `prefix` (the reused
  accumulated checkpoint = all previous prompt+completion tokens), `system`
  / `user` / `tool` (one segment per appended role), `generation_prompt`.
  Segments tile `prompt_token_ids` exactly; this is the loss-mask
  construction material — no re-tokenization needed downstream.
- `prefix_stable` — true iff this turn's `prompt_token_ids` extend the
  previous **recorded** line's `prompt_token_ids + completion_token_ids`.
  The baseline is the record stream itself, not the in-memory checkpoint: a
  rollback applied for a turn that never produced a line (upstream error,
  timeout, 409) surfaces as `false` on the next recorded turn. `false`
  (retry rollback / history rewrite) means the turn must not be spliced
  into one linear token stream.
- `turn_index` — assigned by the sink, monotonic per session, and advances
  even when a line fails to persist: a write/validation failure is logged
  and leaves a detectable index gap instead of an undetectable missing turn.
- `render_skew` (vLLM only) — a cheap per-turn probe comparing the render
  endpoint's from-scratch ids against the gateway's accumulated prompt ids
  (recorded, never enforced); `null` on sglang.

Strictness guarantees: every line is strict JSON (`allow_nan=False` — never
`NaN`/`Infinity` literals); token ids are Python ints and logprobs finite
floats, validated at the sink — a violating record is dropped with a logged
error and an index gap, never silently coerced.

Closing a session appends one final `tito.session.v1` metadata line —
`{"schema_version": "tito.session.v1", "session_id", "turns", "reason":
"deleted" | "ttl_evicted" | "capacity_evicted" | "shutdown", "ts"}` — and
closes the file. Without `--record-dir` nothing is written and the in-memory
behavior is unchanged.

## Session lifecycle

Sessions live in memory until deleted. The intended rollout flow is
**harvest, then delete**: run the rollout, `GET /sessions/{id}` to read the
records + accumulated ids (and the mismatch audit), then
`DELETE /sessions/{id}` — the delete finalizes the record file and frees the
in-memory trajectory and its pool pin.

For long-running gateways two optional guards bound memory:
`--session-ttl-seconds` evicts sessions idle beyond the TTL, and
`--max-sessions` LRU-evicts beyond a count. Eviction always finalizes the
session's record file first and **never** touches a session with an in-flight
request; capacity may transiently overflow rather than kill a live rollout.

Interleaved turns on one session (a second request racing the first) are
rejected with an explicit **409**: the losing turn's completion cannot be
committed to a trajectory that changed under it, and silently serving an
unrecorded response would be capture data loss. Callers retry on the current
session state.

## HTTP surface

- `POST /sessions` → `{session_id}`
- `POST /sessions/{id}/v1/chat/completions` — proxied chat completion; the
  gateway forces the token-recording fields (non-streaming + logprobs), sends
  the pretokenized prompt IDs via the backend kind's dialect, and appends a
  token-aligned checkpoint.
- `GET /sessions/{id}` — records + metadata, incl. `accumulated_token_ids` and
  `tito_session_mismatch` (empty list ⇒ byte-identical to a fresh render).
- `DELETE /sessions/{id}` — close the session and forget its pool pin.

Multiple backend replicas are supported via `BackendPool`: requests are pinned
sticky-by-`session_id` for prefix-cache locality, and a replica is marked down
on a transport error.

## Python API

```python
from agentix.tito import TITOGateway, TITOGatewayConfig

TITOGateway(TITOGatewayConfig(
    hf_checkpoint="Qwen/Qwen3-4B",
    backend_url="http://127.0.0.1:30000",
    tito_model="qwen3",
)).run()
```

`agentix.tito.get_tito_tokenizer(tokenizer, "qwen3")` builds the engine
tokenizer directly if you only want incremental pretokenization.

## Layout

```text
agentix/tito/
├── gateway.py / server.py / pool.py / discovery.py / config.py / cli.py
└── engine/                      — the native TITO token-alignment engine
    ├── pretokenize.py           — TITOTokenizer (+ Qwen3TITOTokenizer)
    ├── compare.py               — special-token-segment mismatch audit
    ├── trajectory.py            — LinearTrajectory + SessionRegistry
    ├── session_app.py           — FastAPI session routes
    ├── upstream.py              — backend-kind adapters (sglang | vllm)
    ├── messages.py / render.py / processing.py / errors.py
    └── templates/qwen3_fixed.jinja
```

## Tests

```bash
pytest plugins/tito/tests
```

The engine tests are self-contained — they build a tiny in-memory tokenizer, so
no model download or GPU is required.
