# 开发说明

[文档首页](index.zh-CN.md) | [English](development.md)

## 源码策略

这个 package 是对 Miles 工作的公开封装。实现上应尽可能复用 upstream 逻辑。

维护规则：

- 保留 Miles attribution 和 upstream notices。
- 持续更新 vendored source audit。
- 优先写 wrapper，不改 TITO 算法。
- copied upstream tests 是兼容性合同。
- 不弱化 negative tests。

## 公开文档标准

公开文档需要讲清楚关系：

- 底层 TITO 设计和实现来自 Miles。
- 这个 package 让这套工作可以作为 standalone gateway 被 import 和运行。
- optional heavy verification 依赖 Miles/SGLang training stack。

## 本地设置

Editable install 是给仓库开发用的：

```bash
pip install -e '.[test]'
pytest tests/package -q
```

公开用户应该安装 package distribution：

```bash
pip install tito-gateway
pip install 'tito-gateway[verify]'
```

## 构建 Distribution

```bash
python -m pip install build
python -m build
python -m pip install dist/tito_gateway-0.1.0-py3-none-any.whl
tito-gateway --help
```

Release tag 使用 `vX.Y.Z`。GitHub Actions workflow 会在每次 push 时构建，在 version
tag 上通过 PyPI trusted publishing 发布到 `pypi` environment。

## Push 前检查

1. 跑 CPU-fast tests。
2. 跑 CLI help smoke checks。
3. 确认 staged files 不包含 cache、模型权重、credentials 或本地 env 文件。
4. 对 staged 和 outgoing changes 做 secret scan。
5. 检查 `git diff --check`。

## 当前 Secret Guard

push 前可以使用 `guard-secret` skill。安装后，在 `git push` 前运行它；只有报告
`SAFE_TO_PUSH` 时才继续 push。
