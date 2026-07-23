"""Upstream backend adapters — one exact-token chat turn per backend kind.

The session routes are backend-dialect-agnostic: an adapter owns everything
that differs between inference servers — which fields to force on the agent's
chat body, which endpoint(s) carry the pretokenized prompt ids, and where the
exact completion token ids come back. Two kinds ship:

- ``sglang`` — one chat-completions call; ``input_ids`` carries the prompt,
  ``meta_info.output_token_logprobs`` carries the completion ids.
- ``vllm`` (>= 0.24.0) — chat completions cannot take pretokenized input (an
  sglang-style ``input_ids`` field is silently ignored: the request "succeeds"
  while the server re-tokenizes ``messages`` itself), so the turn is the
  token-native three-call chain: ``/v1/chat/completions/render`` translates
  the chat request into a ``GenerateRequest`` (resolving sampling params
  server-side; its from-scratch ``token_ids`` are discarded),
  ``/inference/v1/generate`` runs on the session's exact prompt ids, and
  ``/v1/chat/completions/derender`` turns the raw token output back into a
  chat completion via the server-side tool/reasoning parsers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from fastapi import Request
from starlette.responses import Response

from .errors import MessageValidationError, UpstreamResponseError

BACKEND_KINDS = ("sglang", "vllm")


class Backend(Protocol):
    async def do_proxy(self, request: Request, path: str, body: bytes | None = None) -> dict: ...
    def build_proxy_response(self, result: dict) -> Response: ...


@dataclass(frozen=True)
class TurnHarvest:
    """The token-exact outcome of a successful chat turn.

    ``completion_logprobs`` pairs 1:1 with ``completion_token_ids`` — the
    sampled-token logprobs both backends are forced to return (they are the
    per-token cross-check AND the recorded rollout logprobs, so they are
    retained here, never validated-then-discarded). ``render_token_ids`` is
    vLLM-only: the render endpoint's from-scratch prompt ids, kept for the
    cheap per-turn skew probe against the gateway's accumulated prompt ids.
    """

    response: dict
    assistant_message: dict
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    finish_reason: str | None = None
    render_token_ids: list[int] | None = None


@dataclass(frozen=True)
class ChatTurn:
    """``proxy_result`` is always what the agent receives; ``harvest`` is None
    when the upstream answered non-200 (pass through, record nothing)."""

    proxy_result: dict
    harvest: TurnHarvest | None = None


class UpstreamAdapter(Protocol):
    kind: str

    def validate_request(self, request_body: dict) -> None:
        """Raise for request-shape preconditions. Called BEFORE the session
        lock: a rejected request must be a pure 4xx with no committed rollback
        side effects."""
        ...

    async def chat_turn(
        self, backend: Backend, request: Request, request_body: dict, prompt_token_ids: list[int]
    ) -> ChatTurn: ...


def _encode(obj: dict) -> bytes:
    return json.dumps(obj).encode()


def _parse_response_object(body: bytes) -> dict:
    # Structural failures in a 200 body are the backend's fault — surface
    # every shape violation as a clean 502, never an unhandled 500.
    try:
        response = json.loads(body)
    except ValueError as e:
        raise UpstreamResponseError(f"backend returned an unparseable 200 body: {e}") from e
    if not isinstance(response, dict):
        raise UpstreamResponseError("backend 200 body is not a JSON object")
    return response


def _first_choice(response: dict) -> dict:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise UpstreamResponseError("backend 200 body has no choices")
    return choices[0]


def _extract_assistant_message(choice: dict) -> dict:
    assistant_message = choice.get("message")
    if not isinstance(assistant_message, dict):
        raise UpstreamResponseError("assistant message missing")
    if assistant_message.get("content") is None and not assistant_message.get("tool_calls"):
        # Tool-call-only turns routinely carry content:null (the parser
        # consumed all generated text) — only a turn with NEITHER content
        # NOR tool_calls is malformed.
        raise UpstreamResponseError("assistant message has neither content nor tool_calls")
    return assistant_message


class SglangUpstream:
    """One chat-completions call; sglang's ``input_ids`` + ``meta_info``
    extensions carry the exact tokens both ways."""

    kind = "sglang"

    def validate_request(self, request_body: dict) -> None:
        """No request-shape preconditions beyond the common body checks."""

    async def chat_turn(
        self, backend: Backend, request: Request, request_body: dict, prompt_token_ids: list[int]
    ) -> ChatTurn:
        # Hardcoded so an agent override can't break token accumulation:
        request_body["logprobs"] = True          # -> meta_info.output_token_logprobs
        request_body["return_meta_info"] = True   # -> choice.meta_info
        request_body["no_stop_trim"] = False       # stop-token text trimmed from content
        # The TITO flow needs the complete JSON completion (logprobs +
        # meta_info); an SSE stream would be unparseable below. Force
        # non-streaming — a stream:true agent gets the full JSON body back.
        request_body["stream"] = False
        request_body.pop("stream_options", None)
        request_body["input_ids"] = prompt_token_ids

        result = await backend.do_proxy(request, "v1/chat/completions", body=_encode(request_body))
        if result["status_code"] != 200:
            return ChatTurn(proxy_result=result)

        response = _parse_response_object(result["response_body"])
        choice = _first_choice(response)
        meta_info = choice.get("meta_info")
        if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
            raise UpstreamResponseError("meta_info.output_token_logprobs missing (needs logprobs=True)")
        assistant_message = _extract_assistant_message(choice)
        output_token_logprobs = meta_info["output_token_logprobs"]
        completion_tokens = meta_info.get("completion_tokens")
        if not isinstance(output_token_logprobs, list) or not isinstance(completion_tokens, int):
            raise UpstreamResponseError("meta_info output_token_logprobs/completion_tokens malformed")
        if len(output_token_logprobs) != completion_tokens:
            raise UpstreamResponseError(
                f"len(output_token_logprobs)={len(output_token_logprobs)} != completion_tokens={completion_tokens}"
            )
        try:
            # sglang entries are [logprob, token_id, text].
            completion_token_ids = [t[1] for t in output_token_logprobs]
            completion_logprobs = [float(t[0]) for t in output_token_logprobs]
        except (TypeError, ValueError, IndexError, KeyError) as e:
            raise UpstreamResponseError(f"malformed output_token_logprobs entry: {e}") from e

        return ChatTurn(
            proxy_result=result,
            harvest=TurnHarvest(
                response=response,
                assistant_message=assistant_message,
                completion_token_ids=completion_token_ids,
                completion_logprobs=completion_logprobs,
                finish_reason=_finish_reason(choice),
            ),
        )


def _rewrite_tool_call_finish_reasons(response: dict) -> bool:
    """vLLM's derender passes ``finish_reason`` through verbatim ("stop"),
    while its own chat endpoint reports "tool_calls" when the parser found
    calls — rewrite for drop-in parity so agent loops that branch on
    finish_reason behave identically on both backend kinds."""
    changed = False
    choices = response.get("choices")
    for choice in choices if isinstance(choices, list) else []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and message.get("tool_calls") and choice.get("finish_reason") == "stop":
            choice["finish_reason"] = "tool_calls"
            changed = True
    return changed


def _require_model(request_body: dict) -> str:
    model = request_body.get("model")
    if not isinstance(model, str) or not model:
        raise MessageValidationError(
            "the vllm backend requires 'model' in the chat request (derender rejects a missing model)"
        )
    return model


class VllmUpstream:
    """render -> generate -> derender against vLLM >= 0.24.0 (the first
    release with the derender endpoints)."""

    kind = "vllm"

    def validate_request(self, request_body: dict) -> None:
        _require_model(request_body)

    async def chat_turn(
        self, backend: Backend, request: Request, request_body: dict, prompt_token_ids: list[int]
    ) -> ChatTurn:
        model = _require_model(request_body)
        # Hardcoded so an agent override can't break token accumulation.
        # logprobs ride along as the per-token cross-check + recorded rollout
        # logprobs; render maps sampling_params.logprobs from top_logprobs, so
        # pin it to 0 (sampled token only) unless the agent asked for more.
        # An explicit top_logprobs:null must be coerced too — openai-python
        # serializes an explicitly-passed None, and null would make render
        # resolve sampling logprobs to null (no logprobs block from generate).
        request_body["logprobs"] = True
        if request_body.get("top_logprobs") is None:
            request_body["top_logprobs"] = 0
        # derender only accepts a complete (non-streamed) GenerateResponse.
        request_body["stream"] = False
        request_body.pop("stream_options", None)

        render_result = await backend.do_proxy(
            request, "v1/chat/completions/render", body=_encode(request_body)
        )
        if render_result["status_code"] != 200:
            return ChatTurn(proxy_result=render_result)
        generate_request = _parse_response_object(render_result["response_body"])
        # Keep render's from-scratch ids for the recorded skew probe before
        # they are discarded from the generate request below.
        rendered = generate_request.get("token_ids")
        render_token_ids = (
            list(rendered) if isinstance(rendered, list) and all(isinstance(t, int) for t in rendered) else None
        )
        # The whole point: generate from the session's accumulated prompt ids,
        # not render's from-scratch re-render of the message history.
        generate_request["token_ids"] = prompt_token_ids
        # render copies `stream` from the chat request it saw — re-force.
        generate_request["stream"] = False
        generate_request.pop("stream_options", None)

        generate_result = await backend.do_proxy(
            request, "inference/v1/generate", body=_encode(generate_request)
        )
        if generate_result["status_code"] != 200:
            return ChatTurn(proxy_result=generate_result)
        generate_response = _parse_response_object(generate_result["response_body"])
        completion_token_ids, completion_logprobs = _harvest_generate_tokens(generate_response)

        derender_request = {
            "model": model,
            "generate_response": generate_response,
            # The released GenerateResponse carries no usage; derender
            # defaults prompt_tokens to 0 unless the caller supplies it.
            "prompt_tokens": len(prompt_token_ids),
            # The tool/reasoning parsers read tools + tool_choice from the
            # original chat request; without it derender falls back to plain
            # detokenization into content.
            "chat_request": request_body,
        }
        derender_result = await backend.do_proxy(
            request, "v1/chat/completions/derender", body=_encode(derender_request)
        )
        if derender_result["status_code"] != 200:
            return ChatTurn(proxy_result=derender_result)
        response = _parse_response_object(derender_result["response_body"])
        assistant_message = _extract_assistant_message(_first_choice(response))
        if assistant_message.get("reasoning") is not None and assistant_message.get("reasoning_content") is None:
            # The engine's templates and the mismatch audit read the
            # sglang/DeepSeek `reasoning_content` key; mirror it on the stored
            # trajectory copy only — the wire response keeps vLLM's
            # `reasoning` verbatim.
            assistant_message = {**assistant_message, "reasoning_content": assistant_message["reasoning"]}
        if _rewrite_tool_call_finish_reasons(response):
            derender_result = {**derender_result, "response_body": _encode(response)}

        return ChatTurn(
            proxy_result=derender_result,
            harvest=TurnHarvest(
                response=response,
                assistant_message=assistant_message,
                completion_token_ids=completion_token_ids,
                completion_logprobs=completion_logprobs,
                # After the tool-call rewrite: the recorded finish_reason is
                # what the agent actually received.
                finish_reason=_finish_reason(_first_choice(response)),
                render_token_ids=render_token_ids,
            ),
        )


def _finish_reason(choice: dict) -> str | None:
    reason = choice.get("finish_reason")
    return reason if isinstance(reason, str) else None


def _harvest_generate_tokens(generate_response: dict) -> tuple[list[int], list[float]]:
    """Completion (token_ids, logprobs) from a generate response — logprobs
    are the sampled-token values from the SAME forward pass, kept for the
    turn record, not just validated."""
    choice = _first_choice(generate_response)
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids or not all(isinstance(t, int) for t in token_ids):
        raise UpstreamResponseError("generate response token_ids missing or malformed")
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        # logprobs were forced on; their absence means the forcing was ignored
        # (wrong server / stripped by a proxy) — refuse rather than record a
        # turn without the per-token cross-check.
        raise UpstreamResponseError("generate response logprobs missing (needs logprobs forced on)")
    if len(content) != len(token_ids):
        raise UpstreamResponseError(
            f"len(logprobs.content)={len(content)} != len(token_ids)={len(token_ids)}"
        )
    try:
        completion_logprobs = [float(entry["logprob"]) for entry in content]
    except (TypeError, ValueError, KeyError) as e:
        raise UpstreamResponseError(f"malformed logprobs.content entry: {e}") from e
    return list(token_ids), completion_logprobs


def get_upstream(kind: str) -> UpstreamAdapter:
    if kind == "sglang":
        return SglangUpstream()
    if kind == "vllm":
        return VllmUpstream()
    raise ValueError(f"unsupported backend_kind {kind!r}; supported: {list(BACKEND_KINDS)}")
