# Golden parity fixtures

Each subdirectory holds one direction of translation:

- `requests/`  — input `anthropic_<name>.json`, expected output `openai_<name>.json`.
- `responses/` — input `openai_<name>.json`, expected output `anthropic_<name>.json`. Each file pair also carries a `meta_<name>.json` with the `original_model` and the `tool_map` (typically empty) that the response translator needs.
- `streams/`   — input `openai_<name>.sse`, expected output `anthropic_<name>.jsonl` (one Anthropic event per line, in order).

Goldens were produced by:

1. **LiteLLM (primary oracle)**: `python scripts/regen_fixtures.py` (requires `pip install 'litellm>=1.0'`). Where LiteLLM and 1rgs/claude-code-proxy disagree, LiteLLM wins; the divergence is noted in `source.txt`.

2. **Hand-curated**: the streams and responses are typically hand-crafted from the OpenAI Chat Completions API reference, because LiteLLM's response side does not have a simple "translate one chunk" entry point.

To regenerate goldens after a rule change:

```bash
pip install 'litellm>=1.0'
python scripts/regen_fixtures.py
```
