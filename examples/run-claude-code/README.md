# run-claude-code

Minimal end-to-end example:

```text
Claude Code inside Docker sandbox
  -> ANTHROPIC_BASE_URL=http://127.0.0.1:<port>
  -> sandbox-local abridge-mitm
  -> Agentix runtime channel
  -> host OpenAIForwarder
  -> OpenAI-compatible upstream
```

The upstream API key stays on the host. The sandbox only receives a
local Anthropic-shaped URL and a dummy Anthropic key. The transport
does not require Docker-specific host networking.

## Build

Claude Code is currently packaged as an x86_64 Linux binary in the
Agentix plugin, so build and run this example as `linux/amd64`.

```bash
cd examples/run-claude-code
uv sync
uv run agentix build . --name run-claude-code:0.1.0 --platform linux/amd64 --format oci-image
```

## Run

Use any OpenAI-compatible endpoint:

```bash
OPENAI_BASE_URL=https://example.com/v1 \
OPENAI_API_KEY=sk-... \
OPENAI_MODEL=your-model \
uv run python main.py --bundle run-claude-code:0.1.0
```

DashScope compatible-mode example:

```bash
ABRIDGE_OPENAI_EXTRA_BODY='{"enable_thinking":true}' \
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
OPENAI_API_KEY=sk-... \
OPENAI_MODEL=qwen3.7-max \
uv run python main.py --bundle run-claude-code:0.1.0
```

Optional provider-specific fields can be passed without changing the
code:

```bash
ABRIDGE_OPENAI_EXTRA_BODY='{"enable_thinking":true}' \
OPENAI_BASE_URL=https://example.com/v1 \
OPENAI_API_KEY=sk-... \
OPENAI_MODEL=your-model \
uv run python main.py --bundle run-claude-code:0.1.0
```

The example creates a tiny Git repo in the sandbox with
`math_utils.py`, asks Claude Code to add `add(a, b)`, and prints the
resulting diff plus a Python verification.
