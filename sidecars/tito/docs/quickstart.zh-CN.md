# 快速开始

[文档首页](index.zh-CN.md) | [English](quickstart.md)

## 安装

```bash
pip install tito-gateway
```

Miles TITO 逻辑已经 copy/vendor 在这个 package 里。只有需要在本地跑 heavy verifier
路径时，才安装 verifier 可选依赖：

```bash
pip install 'tito-gateway[verify]'
```

如果 console script 不在 `PATH` 中，可以用：

```bash
python -m tito_gateway.cli --help
```

## 使用显式 backend 启动

```bash
tito-gateway --hf-checkpoint Qwen/Qwen3-0.6B \
  --tito-model qwen3 \
  --tito-allowed-append-roles tool user \
  --backend-url http://127.0.0.1:8000 \
  --session-server-port 30000
```

默认命令就是 `serve`，所以 `tito-gateway ...` 和 `tito-gateway serve ...` 走同一套
启动逻辑。

## 使用 backend probing 启动

```bash
tito-gateway serve --hf-checkpoint Qwen/Qwen3-0.6B \
  --backend-probe-candidate http://127.0.0.1:8000 \
  --backend-probe-candidate http://127.0.0.1:30000 \
  --backend-probe-timeout 0.5
```

## 在 Python 中嵌入

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

## 运行 CPU-fast 测试

```bash
pip install -e '.[test]'
python scripts/prepare_test_tokenizer_cache.py --endpoint https://huggingface.co
pytest tests/upstream tests/package -q
```
