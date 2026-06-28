# 核心概念

[文档首页](index.zh-CN.md) | [English](concepts.md)

## 运行模型

Gateway 有三部分：

1. 一个 OpenAI-compatible backend。
2. 一个封装 Miles session server 的 FastAPI gateway。
3. 客户端先创建 session，再通过 session routes 发送 chat completions。

典型流程：

1. 启动或发现 backend。
2. 用 checkpoint 和 TITO 配置启动 gateway。
3. 通过 `POST /sessions` 创建 session。
4. 通过 `/sessions/{session_id}/v1/chat/completions` 发送 chat 请求。
5. Gateway 注入 session/TITO metadata，并从 backend response metadata 更新 token 状态。

## Session 不变量

Miles 的 TITO 路径依赖 append-only message history。例外是 Miles 支持的围绕最新
assistant checkpoint 的 rollback。

`tito_allowed_append_roles` 声明 assistant turn 后允许追加的角色。默认 role surface 是
`tool`。

## Backend 自动发现

Backend 选择是 deterministic 的：

1. 显式配置或 `--backend-url`。
2. 环境变量：
   - `TITO_BACKEND_URL`
   - `OPENAI_BASE_URL`
   - `SGLANG_BASE_URL`
3. 配置的 probe candidates，按顺序。

每个 probe candidate 先检查 `/health`，再检查 `/v1/models`。第一个 live candidate
会被选中。如果找不到 backend，gateway 会在绑定前失败。

## Session Routes

- `GET /health`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /sessions/{session_id}/v1/chat/completions`

Chat completion 请求会被 proxy 到 backend。proxy 路径会注入 Miles 实现所需字段，包括
token IDs 和 metadata requests。Backend response 必须包含 output token logprob
metadata，session trajectory 才能更新。
