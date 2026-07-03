"""FastAPI session routes for the TITO gateway.

The gateway keeps a token-aligned trajectory per session and proxies chat
completions to an OpenAI-compatible backend. The chat-completions flow:
prepare pretokenized prompt ids (lock held briefly) -> run the backend turn
via the kind-specific upstream adapter (no lock; see ``engine.upstream``) ->
append the trajectory checkpoint (lock held briefly). The upstream exchange is
NOT held under the lock so a slow generation doesn't block DELETE/other ops.

`setup_session_routes` is transport-agnostic: pass any object exposing
``do_proxy(request, path, body=None) -> dict`` and ``build_proxy_response(result)``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from .errors import (
    MessageValidationError,
    SessionError,
    SessionNotFoundError,
    TokenizationError,
)
from .pretokenize import get_tito_tokenizer
from .processing import load_tokenizer
from .trajectory import GetSessionResponse, SessionRecord, SessionRegistry
from .upstream import Backend, get_upstream

logger = logging.getLogger(__name__)


def build_registry(args: Any) -> SessionRegistry | None:
    """Construct a SessionRegistry from gateway args, or None if no hf_checkpoint."""
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] no hf_checkpoint set — session routes disabled")
        return None
    tokenizer = load_tokenizer(
        hf_checkpoint,
        chat_template_path=getattr(args, "chat_template_path", None),
        # Executes Python shipped inside the checkpoint repo — explicit opt-in
        # (config/--trust-remote-code), never a hardcoded default.
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
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
    # Exposed for operational introspection (session counts, tests).
    app.state.tito_registry = registry

    adapter = get_upstream(getattr(args, "backend_kind", "sglang"))

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
        # `closing` is only ever set INSIDE the lock, immediately before the
        # removal: a DELETE cancelled while waiting for the lock (client
        # timeout/disconnect) must leave the session deletable, not wedged
        # behind a stuck flag with the trajectory leaked in the registry.
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            session.closing = True
            registry.remove_session(session_id)
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str) -> Response:
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

        # Read + parse the body BEFORE taking the lock: the read lasts as long
        # as the client's upload — under the lock, one dribbling client would
        # wedge DELETE and every other operation on the session.
        raw = await request.body()
        try:
            request_body = json.loads(raw) if raw else {}
        except ValueError as e:
            raise MessageValidationError(f"request body is not valid JSON: {e}") from e
        if not isinstance(request_body, dict):
            raise MessageValidationError("request body must be a JSON object")

        # Phase 1: prepare the pretokenized prompt ids (lock held briefly).
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            request_messages = request_body.get("messages", [])
            prompt_token_ids = session.prepare_pretokenized(
                request_messages, tools=request_body.get("tools"), tito_tokenizer=registry.tito_tokenizer
            )
            # `version` advances on BOTH update and rollback; `num_assistant`
            # would be ABA-prone (a concurrent rollback+update restores it).
            expected_version = session.version

        # Phase 2: run the backend turn (NO lock). The adapter forces the
        # token-recording fields onto the body (so an agent override can't
        # break token accumulation) and harvests the exact completion ids.
        turn = await adapter.chat_turn(backend, request, request_body, prompt_token_ids)
        if turn.harvest is None:
            return backend.build_proxy_response(turn.proxy_result)
        harvest = turn.harvest

        # Phase 3: append the trajectory checkpoint (lock held briefly).
        async with session.lock:
            if session.closing:
                return backend.build_proxy_response(turn.proxy_result)
            if session.version != expected_version:
                logger.warning("session %s changed during proxy; skipping state update", session_id)
                return backend.build_proxy_response(turn.proxy_result)
            session.update_pretokenized_state(
                request_messages,
                harvest.assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=harvest.completion_token_ids,
                max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
            )
            session.append_record(
                SessionRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path="/v1/chat/completions",
                    status_code=turn.proxy_result["status_code"],
                    request=request_body,
                    response=harvest.response,
                )
            )
        return backend.build_proxy_response(turn.proxy_result)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str) -> Response:
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
