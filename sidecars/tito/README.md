# TITO Gateway

[English](README.md) | [中文](README.zh-CN.md)

TITO Gateway is a standalone Python package and CLI wrapper around the Miles
Agentic Chat Template / TITO session-server work. It is a public-facing
packaging and usage layer for Miles' TITO implementation, not a rewrite of the
underlying algorithms.

The TITO tokenizer, fixed chat-template approach, session trajectory model,
proxy behavior, and verifier flow are credited to Miles and its contributors.
Vendored source is tracked in `tito_gateway/VENDORED_MILES_AUDIT.md`.

## Documentation

- [Docs Home](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Concepts](docs/concepts.md)
- [Python API](docs/api.md)
- [CLI Reference](docs/cli.md)
- [Verification And Tests](docs/verification.md)
- [Development Notes](docs/development.md)
- [中文文档入口](docs/index.zh-CN.md)

## Quickstart

Install from PyPI:

```bash
pip install tito-gateway
```

The copied Miles TITO logic is bundled in this package. Install optional
verifier dependencies only when you want to run the heavy verifier path locally:

```bash
pip install 'tito-gateway[verify]'
```

Start the gateway beside an OpenAI-compatible backend:

```bash
tito-gateway --hf-checkpoint Qwen/Qwen3-0.6B \
  --tito-model qwen3 \
  --tito-allowed-append-roles tool user \
  --backend-url http://127.0.0.1:8000 \
  --session-server-port 30000
```

Or embed it in Python:

```python
from tito_gateway import TITOGateway, TITOGatewayConfig

gateway = TITOGateway(
    TITOGatewayConfig(
        hf_checkpoint="Qwen/Qwen3-0.6B",
        backend_url="http://127.0.0.1:8000",
        tito_model="qwen3",
        tito_allowed_append_roles=("tool", "user"),
    )
)

app = gateway.app
```

Run the CPU-fast test suite:

```bash
pip install -e '.[test]'
pytest tests/upstream tests/package -q
```

## Acknowledgement

This package explicitly acknowledges Miles as the source of the underlying TITO
design and implementation:

- Miles repository: https://github.com/radixark/miles
- Miles documentation: https://www.radixark.com/miles/docs/user-guide/agentic-chat-template

Future vendoring should preserve upstream notices, keep the source audit
current, and keep copied upstream tests as the compatibility contract.
