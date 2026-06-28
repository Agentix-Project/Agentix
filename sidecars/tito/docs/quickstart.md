# Quickstart

[Docs Home](index.md) | [中文](quickstart.zh-CN.md)

## Install

```bash
pip install tito-gateway
```

The copied Miles TITO logic is bundled in this package. Install optional
verifier dependencies only when you want to run the heavy verifier path locally:

```bash
pip install 'tito-gateway[verify]'
```

If the console script is not on `PATH`, use:

```bash
python -m tito_gateway.cli --help
```

## Start With An Explicit Backend

```bash
tito-gateway --hf-checkpoint Qwen/Qwen3-0.6B \
  --tito-model qwen3 \
  --tito-allowed-append-roles tool user \
  --backend-url http://127.0.0.1:8000 \
  --session-server-port 30000
```

The default command is `serve`, so `tito-gateway ...` and `tito-gateway serve ...`
use the same startup path.

## Start With Backend Probing

```bash
tito-gateway serve --hf-checkpoint Qwen/Qwen3-0.6B \
  --backend-probe-candidate http://127.0.0.1:8000 \
  --backend-probe-candidate http://127.0.0.1:30000 \
  --backend-probe-timeout 0.5
```

## Embed In Python

```python
from tito_gateway import TITOGateway, TITOGatewayConfig

gateway = TITOGateway(
    TITOGatewayConfig(
        hf_checkpoint="Qwen/Qwen3-0.6B",
        backend_url="http://127.0.0.1:8000",
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="qwen3",
        tito_allowed_append_roles=("tool", "user"),
    )
)

app = gateway.app
```

## Run CPU-Fast Tests

```bash
pip install -e '.[test]'
python scripts/prepare_test_tokenizer_cache.py --endpoint https://huggingface.co
pytest tests/upstream tests/package -q
```
