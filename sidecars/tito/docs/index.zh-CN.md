# TITO Gateway 文档

[English](index.md) | [中文](index.zh-CN.md)

TITO Gateway 是围绕 Miles Agentic Chat Template / TITO session-server 工作做的
公开 package 和 CLI 封装。它复用 Miles 逻辑，并把这条路径整理成独立 package。

## 从这里开始

- [快速开始](quickstart.zh-CN.md)：安装、运行和测试。
- [核心概念](concepts.zh-CN.md)：运行模型、session flow 和不变量。
- [Python API](api.zh-CN.md)：把 gateway 嵌入到 backend 旁边。
- [CLI 参考](cli.zh-CN.md)：serve 命令和 backend discovery。
- [验证与测试](verification.zh-CN.md)：verifier 命令和 CPU-fast 测试准备。
- [开发说明](development.zh-CN.md)：源码策略、致谢和发布检查。

## 公开致谢

本 package 明确致谢 Miles：底层 TITO 实现来自 Miles。这个项目是封装和使用层，不是
重新实现。源码审计记录在 `tito_gateway/VENDORED_MILES_AUDIT.md`。
