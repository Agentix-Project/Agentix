# Anthropic ↔ OpenAI conversion fixtures

Golden parity fixtures for bidirectional Anthropic Messages ↔ OpenAI
Chat Completions translation, exercised by
`tests/test_anthropic_transform_fixtures.py` against
`agentix.bridge.clients._anthropic_transforms`.

## Origin and license

These files are copied verbatim from the test corpus of
[cc_convert](https://github.com/yitianlian/cc_convert) (a Rust
Anthropic ↔ OpenAI protocol converter), specifically from
`tests/fixtures/` as vendored on this repo's `abridge/gateway-tito`
ref (`sidecars/cc_convert/tests/fixtures/`). Only the fixture data was
copied — no Rust code, no scripts.

cc_convert is dual-licensed **MIT OR Apache-2.0**; we redistribute the
fixtures under the MIT license. The upstream MIT license text is
included alongside this file as [`LICENSE-MIT`](LICENSE-MIT)
(Copyright (c) 2026 cc_convert maintainers).

## Layout

Each subdirectory holds one direction of translation. Files pair up by
the shared `<NN>_<case_name>` suffix:

- `requests/` — Anthropic → OpenAI request translation.
  Input `anthropic_<case>.json`, expected output `openai_<case>.json`,
  plus `tool_map_<case>.json` recording any sanitized-tool-name
  mapping (empty for all cases except `13_long_tool_name`).
- `responses/` — OpenAI → Anthropic response translation.
  Input `openai_<case>.json`, expected output `anthropic_<case>.json`,
  plus `meta_<case>.json` carrying the `original_model` and the
  `tool_map` the reference translator was invoked with.
- `streams/` — OpenAI SSE → Anthropic event-stream translation.
  Input `openai_<case>.sse` (Chat Completions chunks), expected output
  `anthropic_<case>.jsonl` (one Anthropic stream event per line, in
  order).

## Provenance of the goldens

Upstream generated the request goldens with LiteLLM's
`AnthropicAdapter` as the primary parity oracle (cross-referenced
against 1rgs/claude-code-proxy and maxnowack/anthropic-proxy); the
response and stream goldens are hand-curated from the OpenAI Chat
Completions API reference. Some goldens therefore carry LiteLLM
serialization artifacts (explicit `"thinking_blocks": null`,
`"provider_specific_fields": null`, an unconditional empty leading
text block in streams) — the test suite's comparison helpers document
how those are normalized.
