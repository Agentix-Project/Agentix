# Verification And Tests

[Docs Home](index.md) | [中文](verification.zh-CN.md)

## Tokenizer Cache

Copied upstream fast tests need tokenizer assets. Prepare the local cache
without downloading model weights:

```bash
python scripts/prepare_test_tokenizer_cache.py --endpoint https://huggingface.co
```

## CPU-Fast Suite

```bash
pytest tests/upstream tests/package -q
```

Targeted checks:

```bash
pytest tests/upstream/fast/utils/chat_template_utils/test_tito_tokenizer.py
pytest tests/upstream/fast/utils/chat_template_utils/test_pretokenized_via_tito.py
pytest tests/upstream/fast/router/test_sessions.py
pytest tests/upstream/fast/router/test_session_race_conditions.py
pytest tests/upstream/fast/router/test_session_pretokenized_e2e.py
pytest tests/upstream/fast/utils/test_utils/test_session_verify_runner.py
pytest tests/package
```

## CLI Smoke Checks

```bash
tito-gateway --help
tito-gateway serve --help
tito-gateway verify-chat-template --help
tito-gateway verify-session-tito-tokenizer --help
```

## Optional Heavy Verifier

`verify-session-tito-tokenizer` runs the migrated Miles/SGLang session verifier
when the optional training stack is installed. Without that stack, it exits
with a clear dependency or runtime error. That dependency-gated exit is not a
GPU/e2e pass.
