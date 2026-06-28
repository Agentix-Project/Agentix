# 验证与测试

[文档首页](index.zh-CN.md) | [English](verification.md)

## Tokenizer Cache

复制过来的 upstream fast tests 需要 tokenizer assets。可以只准备本地 tokenizer cache，
不下载模型权重：

```bash
python scripts/prepare_test_tokenizer_cache.py --endpoint https://huggingface.co
```

## CPU-Fast Suite

```bash
pytest tests/upstream tests/package -q
```

Targeted checks：

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

`verify-session-tito-tokenizer` 会在安装可选 Miles/SGLang training stack 后运行迁移后的
session verifier。没有这套依赖时，它会以清晰的 dependency 或 runtime error 退出。
这个 dependency-gated exit 不能算作 GPU/e2e pass。
