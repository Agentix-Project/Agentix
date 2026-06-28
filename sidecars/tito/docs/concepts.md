# Concepts

[Docs Home](index.md) | [中文](concepts.zh-CN.md)

## Runtime Model

The gateway has three parts:

1. An OpenAI-compatible backend.
2. A FastAPI gateway that wraps the Miles session server.
3. Clients that create sessions and send chat completions through session routes.

Typical flow:

1. Start or discover a backend.
2. Start the gateway with checkpoint and TITO options.
3. Create a session with `POST /sessions`.
4. Send chat requests through `/sessions/{session_id}/v1/chat/completions`.
5. The gateway injects session/TITO metadata and updates token state from backend response metadata.

## Session Invariants

Miles' TITO path relies on append-only message history. The exception is the
Miles-supported rollback around the latest assistant checkpoint.

`tito_allowed_append_roles` declares which roles may be appended after an
assistant turn. `tool` is the default role surface.

## Backend Discovery

Backend selection is deterministic:

1. Explicit config or `--backend-url`.
2. Environment variables:
   - `TITO_BACKEND_URL`
   - `OPENAI_BASE_URL`
   - `SGLANG_BASE_URL`
3. Configured probe candidates, in order.

Each probe candidate is checked at `/health` first and `/v1/models` second. The
first live candidate wins. If no backend is found, startup fails before binding
the gateway.

## Session Routes

- `GET /health`
- `POST /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /sessions/{session_id}/v1/chat/completions`

Chat completion requests are proxied to the backend. The proxy path injects the
fields expected by the Miles implementation, including token IDs and metadata
requests. Backend responses must include output token logprob metadata so the
session trajectory can update.
