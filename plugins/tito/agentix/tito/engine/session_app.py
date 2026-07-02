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

from .errors import (
    MessageValidationError,
    SessionError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
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

        # Phase 1: prepare pretokenized input_ids (lock held briefly).
        async with session.lock:
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            # Hardcoded so an agent override can't break token accumulation:
            request_body["logprobs"] = True          # -> meta_info.output_token_logprobs
            request_body["return_meta_info"] = True   # -> choice.meta_info
            request_body["no_stop_trim"] = False       # stop-token text trimmed from content
            # The TITO flow needs the complete JSON completion (logprobs +
            # meta_info); an SSE stream would be unparseable below. Force
            # non-streaming — a stream:true agent gets the full JSON body back.
            request_body["stream"] = False
            request_body.pop("stream_options", None)
            request_messages = request_body.get("messages", [])
            prompt_token_ids = session.prepare_pretokenized(
                request_messages, tools=request_body.get("tools"), tito_tokenizer=registry.tito_tokenizer
            )
            request_body["input_ids"] = prompt_token_ids
            body = json.dumps(request_body).encode()
            # `version` advances on BOTH update and rollback; `num_assistant`
            # would be ABA-prone (a concurrent rollback+update restores it).
            expected_version = session.version

        # Phase 2: proxy to the backend (NO lock).
        result = await backend.do_proxy(request, "v1/chat/completions", body=body)
        if result["status_code"] != 200:
            return backend.build_proxy_response(result)

        # Structural failures in a 200 body are the backend's fault — surface
        # every shape violation as a clean 502, never an unhandled 500.
        try:
            response = json.loads(result["response_body"])
        except ValueError as e:
            raise UpstreamResponseError(f"backend returned an unparseable 200 body: {e}") from e
        if not isinstance(response, dict):
            raise UpstreamResponseError("backend 200 body is not a JSON object")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise UpstreamResponseError("backend 200 body has no choices")
        choice = choices[0]
        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
            raise UpstreamResponseError("meta_info.output_token_logprobs missing (needs logprobs=True)")
        assistant_message = choice.get("message")
        if not isinstance(assistant_message, dict):
            raise UpstreamResponseError("assistant message missing")
        if assistant_message.get("content") is None and not assistant_message.get("tool_calls"):
            # Tool-call-only turns routinely carry content:null (the parser
            # consumed all generated text) — only a turn with NEITHER content
            # NOR tool_calls is malformed.
            raise UpstreamResponseError("assistant message has neither content nor tool_calls")
        output_token_logprobs = meta_info["output_token_logprobs"]
        completion_tokens = meta_info.get("completion_tokens")
        if not isinstance(output_token_logprobs, list) or not isinstance(completion_tokens, int):
            raise UpstreamResponseError("meta_info output_token_logprobs/completion_tokens malformed")
        if len(output_token_logprobs) != completion_tokens:
            raise UpstreamResponseError(
                f"len(output_token_logprobs)={len(output_token_logprobs)} != completion_tokens={completion_tokens}"
            )
        try:
            completion_token_ids = [t[1] for t in output_token_logprobs]
        except (TypeError, IndexError, KeyError) as e:
            raise UpstreamResponseError(f"malformed output_token_logprobs entry: {e}") from e

        # Phase 3: append the trajectory checkpoint (lock held briefly).
        async with session.lock:
            if session.closing:
                return backend.build_proxy_response(result)
            if session.version != expected_version:
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
