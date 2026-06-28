# CLI Reference

[Docs Home](index.md) | [中文](cli.zh-CN.md)

## Serve

```bash
tito-gateway serve --hf-checkpoint Qwen/Qwen3-0.6B \
  --tito-model qwen3 \
  --tito-allowed-append-roles tool user \
  --backend-url http://127.0.0.1:8000 \
  --session-server-port 30000
```

The top-level command aliases to `serve`, so the `serve` word can be omitted.

## Template Kwargs

```bash
tito-gateway serve --hf-checkpoint Qwen/Qwen3-0.6B \
  --backend-url http://127.0.0.1:8000 \
  --apply-chat-template-kwargs '{"enable_thinking": false}'
```

## Backend Probe Flags

```bash
tito-gateway serve --hf-checkpoint Qwen/Qwen3-0.6B \
  --backend-probe-candidate http://127.0.0.1:8000 \
  --backend-probe-candidate http://127.0.0.1:30000 \
  --backend-probe-timeout 0.5
```

## Verifiers

```bash
tito-gateway verify-chat-template --template path/to/template.jinja --thinking off
```

```bash
tito-gateway verify-session-tito-tokenizer \
  --hf-checkpoint Qwen/Qwen3-4B \
  --tito-model qwen3 \
  --tito-allowed-append-roles tool user \
  --sglang-reasoning-parser qwen3 \
  --sglang-tool-call-parser qwen25 \
  --rollout-num-gpus-per-engine 1
```

Use `--help` on any subcommand for the full parser surface.
