"""FastAPI session routes for the TITO gateway.

The gateway keeps a token-aligned trajectory per session and proxies chat
completions to an OpenAI-compatible backend (sglang). The chat-completions flow:
prepare pretokenized input_ids (lock held briefly) -> force logprobs/meta_info ->
proxy to the backend (no lock) -> validate -> append the trajectory checkpoint
(lock held briefly). The proxy is NOT held under the lock so a slow generation
doesn't block DELETE/other ops.

`build_session_app` is backend-agnostic: pass any object exposing
``do_proxy(request, path, body=None) -> dict`` and ``build_proxy_response(result)``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .errors import SessionError, SessionNotFoundError, TokenizationError, UpstreamResponseError
from .pretokenize import get_tito_tokenizer
from .processing import load_tokenizer
from .trajectory import GetSessionResponse, SessionRecord, SessionRegistry

logger = logging.getLogger(__name__)


class Backend(Protocol):
    async def do_proxy(self, request: Request, path: str, body: bytes | None = None) -> dict: ...
    def build_proxy_response(self, result: dict) -> Response: ...


def build_registry(args: Any) -> SessionRegistry | None:
    """Construct a SessionRegistry from gateway args, or None if no hf_checkpoint."""
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] no hf_checkpoint set — session routes disabled")
        return None
    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )
    roles = getattr(args, "tito_allowed_append_roles", None) or ("tool",)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        allowed_append_roles=tuple(roles),
    )
    return SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)


def setup_session_routes(app: FastAPI, backend: Backend, args: Any) -> None:
    registry = build_registry(args)
    if registry is None:
        return

    instance_id = getattr(args, "session_server_instance_id", None)

    @app.exception_handler(SessionError)
    async def _session_error_handler(request: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        body: dict[str, Any] = {"status": "ok"}
        if instance_id is not None:
            body["session_server_instance_id"] = instance_id
        return body

    @app.post("/sessions")
    async def create_session() -> dict[str, str]:
        return {"session_id": registry.create_session()}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> GetSessionResponse:
        session = registry.get_session(session_id)
        metadata: dict[str, Any] = {}
        try:
            mismatch = registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("failed to compute tito_session_mismatch for %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = registry.tito_tokenizer.max_trim_tokens
        return GetSessionResponse(session_id=session_id, records=session.records, metadata=metadata)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> Response:
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        await session.lock.acquire()
        try:
            registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str) -> Response:
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # Phase 1: prepare pretokenized input_ids (lock held briefly).
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            raw = await request.body()
            request_body = json.loads(raw) if raw else {}
            # Hardcoded so an agent override can't break token accumulation:
            request_body["logprobs"] = True          # -> meta_info.output_token_logprobs
            request_body["return_meta_info"] = True   # -> choice.meta_info
            request_body["no_stop_trim"] = False       # stop-token text trimmed from content
            request_messages = request_body.get("messages", [])
            prompt_token_ids = session.prepare_pretokenized(
                request_messages, tools=request_body.get("tools"), tito_tokenizer=registry.tito_tokenizer
            )
            request_body["input_ids"] = prompt_token_ids
            body = json.dumps(request_body).encode()
            expected_num_assistant = session.num_assistant

        # Phase 2: proxy to the backend (NO lock).
        result = await backend.do_proxy(request, "v1/chat/completions", body=body)
        if result["status_code"] != 200:
            return backend.build_proxy_response(result)

        response = json.loads(result["response_body"])
        choice = response.get("choices", [{}])[0]
        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
            raise UpstreamResponseError("meta_info.output_token_logprobs missing (needs logprobs=True)")
        assistant_message = choice.get("message", {})
        if assistant_message.get("content") is None:
            raise UpstreamResponseError("assistant message content is None")
        output_token_logprobs = meta_info["output_token_logprobs"]
        completion_tokens = meta_info["completion_tokens"]
        if len(output_token_logprobs) != completion_tokens:
            raise UpstreamResponseError(
                f"len(output_token_logprobs)={len(output_token_logprobs)} != completion_tokens={completion_tokens}"
            )
        completion_token_ids = [t[1] for t in output_token_logprobs]

        # Phase 3: append the trajectory checkpoint (lock held briefly).
        async with session.lock:
            if session.closing:
                return backend.build_proxy_response(result)
            if session.num_assistant != expected_num_assistant:
                logger.warning("session %s changed during proxy; skipping state update", session_id)
                return backend.build_proxy_response(result)
            session.update_pretokenized_state(
                request_messages,
                assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
            )
            session.append_record(
                SessionRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path="/v1/chat/completions",
                    status_code=result["status_code"],
                    request=request_body,
                    response=response,
                )
            )
        return backend.build_proxy_response(result)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str) -> Response:
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
