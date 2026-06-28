# TITO Gateway

[English](README.md) | [中文](README.zh-CN.md)

TITO Gateway 是围绕 Miles Agentic Chat Template / TITO session-server 工作做的
独立 Python package 和 CLI 封装。它是对 Miles TITO 实现的公开封装和使用层，不是
底层算法的重新实现。

TITO tokenizer、fixed chat-template 方案、session trajectory 模型、proxy 行为和验证
流程都来自 Miles 及其贡献者。vendored source 的审计记录在
`tito_gateway/VENDORED_MILES_AUDIT.md`。

## 文档入口

- [中文文档首页](docs/index.zh-CN.md)
- [快速开始](docs/quickstart.zh-CN.md)
- [核心概念](docs/concepts.zh-CN.md)
- [Python API](docs/api.zh-CN.md)
- [CLI 参考](docs/cli.zh-CN.md)
- [验证与测试](docs/verification.zh-CN.md)
- [开发说明](docs/development.zh-CN.md)
- [English Docs Home](docs/index.md)

## 快速开始

从 PyPI 安装：

```bash
pip install tito-gateway
```

Miles TITO 逻辑已经 copy/vendor 在这个 package 里。只有需要在本地跑 heavy verifier
路径时，才安装 verifier 可选依赖：

```bash
pip install 'tito-gateway[verify]'
```

在 OpenAI-compatible backend 旁边启动 gateway：

```bash
tito-gateway --hf-checkpoint Qwen/Qwen3-0.6B \
  --tito-model qwen3 \
  --tito-allowed-append-roles tool user \
  --backend-url http://127.0.0.1:8000 \
  --session-server-port 30000
```

或在 Python 中嵌入：

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

运行 CPU-fast 测试：

```bash
pip install -e '.[test]'
pytest tests/upstream tests/package -q
```

## 致谢

本 package 明确致谢 Miles：底层 TITO 设计和实现来自 Miles 项目。

- Miles repository: https://github.com/radixark/miles
- Miles documentation: https://www.radixark.com/miles/docs/user-guide/agentic-chat-template

后续 vendor Miles 源码时，需要保留 upstream notices，更新 source audit，并保持 copied
upstream tests 作为兼容性合同。
