# cc_convert playground

End-to-end round-trip data for **non-streaming** Anthropic ↔ OpenAI conversion.

## Layout

```
playground/
├── run_roundtrip.py          ← the only script you need to run
├── requests/                 ← source Anthropic requests (input to the pipeline)
│   ├── 02_reasoning_request.json
│   ├── 03_forced_tool.json
│   ├── 05_simple_text.json
│   ├── 07_multi_turn_text.json
│   ├── 08_agent_loop_with_tools.json
│   ├── 09_long_response.json
│   └── 10_parallel_tools_text_only.json
└── runs/<UTC-timestamp>/
    └── <fixture_name>/
        ├── 1_anthropic_request.json   ← source (copied verbatim)
        ├── 2_oai_request.json         ← what cc_convert translated and POSTed
        ├── 3_oai_response.json        ← what the upstream returned (raw)
        ├── 4_anthropic_response.json  ← what cc_convert translated back
        ├── tool_map.json              ← (only if any tool name was truncated)
        ├── meta.json                  ← status / latency / http_status
        └── error.txt                  ← (only on failure)
```

## Pipeline

```
1_anthropic_request.json
        ↓  cc_convert.translate_request()
2_oai_request.json
        ↓  POST → upstream /v1/chat/completions
3_oai_response.json
        ↓  cc_convert.translate_response()
4_anthropic_response.json
```

## Usage

```bash
# Run all 7 fixtures against the upstream
python playground/run_roundtrip.py --upstream http://YOUR_UPSTREAM_HOST:8000

# Override the model name (otherwise probes /v1/models)
python playground/run_roundtrip.py --upstream http://... --model /model

# Only one fixture
python playground/run_roundtrip.py --upstream http://... --only agent_loop
```

## Looking at results

```bash
# See the summary table
cat playground/runs/<timestamp>/_summary.json | jq .

# Look at one fixture's complete 4-file chain
ls playground/runs/<timestamp>/08_agent_loop_with_tools/
cat playground/runs/<timestamp>/08_agent_loop_with_tools/1_anthropic_request.json
cat playground/runs/<timestamp>/08_agent_loop_with_tools/2_oai_request.json
cat playground/runs/<timestamp>/08_agent_loop_with_tools/3_oai_response.json
cat playground/runs/<timestamp>/08_agent_loop_with_tools/4_anthropic_response.json
```

## What each fixture exercises

| Fixture | What it tests |
|---|---|
| `02_reasoning_request` | `thinking.budget_tokens=12000` → `reasoning_effort:"high"` |
| `03_forced_tool` | `tool_choice:{type:"any"}` → OpenAI `"required"` |
| `05_simple_text` | baseline single-turn |
| `07_multi_turn_text` | 5-turn pure-text history |
| `08_agent_loop_with_tools` | 5-turn agent loop: 3 client tools + 2 parallel `tool_use` + 2 `tool_result` round-trip |
| `09_long_response` | 1500-token generation (non-streaming under load) |
| `10_parallel_tools_text_only` | `tool_choice:any` + multiple tools, no history |
