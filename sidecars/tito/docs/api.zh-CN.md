# Python API

[文档首页](index.zh-CN.md) | [English](api.md)

## Import Surface

```python
from tito_gateway import TITOGateway, TITOGatewayConfig, SessionServer, get_tito_tokenizer
```

## Gateway 配置

```python
from tito_gateway import TITOGateway, TITOGatewayConfig

config = TITOGatewayConfig(
    hf_checkpoint="Qwen/Qwen3-0.6B",
    backend_url="http://127.0.0.1:8000",
    chat_template_path=None,
    apply_chat_template_kwargs={"enable_thinking": False},
    tito_model="qwen3",
    tito_allowed_append_roles=("tool", "user"),
    session_server_ip="127.0.0.1",
    session_server_port=30000,
    miles_router_timeout=600.0,
)

gateway = TITOGateway(config)
app = gateway.app
```

## 重要字段

- `hf_checkpoint`：必填 checkpoint 或 model ID，用于 tokenizer loading。
- `backend_url`：显式 backend URL；不填时走 discovery。
- `chat_template_path`：可选 fixed chat template。
- `apply_chat_template_kwargs`：传给 template rendering 的 dict。
- `tito_model`：Miles TITO tokenizer family。
- `tito_allowed_append_roles`：assistant turn 后允许追加的 roles。
- `backend_probe_candidates`：可选 probe URL 列表。
- `backend_probe_timeout`：每个 probe endpoint 的 timeout。

## 直接运行

```python
gateway.run()
```

如果要做 ASGI 组合，建议使用 `gateway.app`，交给自己的 server process mount 或 serve。
