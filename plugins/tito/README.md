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

## Per-turn record persistence

With `--record-dir DIR` (env `TITO_RECORD_DIR`), every committed turn appends
one `tito.record.v1` JSON line to `DIR/<session_id>.jsonl` and flushes it —
the file is complete up to the last committed turn even if the process dies
mid-rollout. Each line carries the exact token truth of one turn:

- `prompt_token_ids` / `completion_token_ids` / `completion_logprobs`
  (sampled-token logprobs from the same forward pass, 1:1 with the ids);
- `prompt_segments` — `{start, end, source}` spans over the prompt ids
  (`render`, `prefix`, per appended role, `generation_prompt`): the material
  a trainer needs to build a loss mask without re-tokenizing anything;
- `prefix_stable` — whether this prompt extends the previous committed
  checkpoint; `false` (retry rollback / history rewrite) means the turn must
  not be spliced into one linear token stream;
- `request_id` (echoed from the caller's `x-request-id` header), `model`,
  `backend_kind`, `finish_reason`, `assistant_message`, and a
  `tokenizer_fingerprint` (`checkpoint` + `chat_template_sha256`) pinning the
  render rules;
- `render_skew` (vLLM only) — a cheap per-turn probe comparing the render
  endpoint's from-scratch ids against the gateway's accumulated prompt ids
  (recorded, never enforced).

Closing a session appends one final `tito.session.v1` metadata line
(`reason`: `deleted` / `ttl_evicted` / `capacity_evicted` / `shutdown`) and
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
