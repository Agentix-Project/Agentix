# Python API

[Docs Home](index.md) | [中文](api.zh-CN.md)

## Import Surface

```python
from tito_gateway import TITOGateway, TITOGatewayConfig, SessionServer, get_tito_tokenizer
```

## Gateway Configuration

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

## Important Fields

- `hf_checkpoint`: required checkpoint or model ID for tokenizer loading.
- `backend_url`: explicit backend URL. If omitted, discovery is used.
- `chat_template_path`: optional fixed chat template.
- `apply_chat_template_kwargs`: JSON-like dict passed into template rendering.
- `tito_model`: Miles TITO tokenizer family.
- `tito_allowed_append_roles`: roles allowed after an assistant turn.
- `backend_probe_candidates`: optional tuple of URLs to probe.
- `backend_probe_timeout`: per-endpoint probe timeout.

## Running Directly

```python
gateway.run()
```

For ASGI composition, prefer `gateway.app` and mount or serve it with your own
server process.
