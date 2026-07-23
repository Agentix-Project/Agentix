"""Per-call request-id propagation between capture and transport layers.

The tunnel deliberately carries no HTTP metadata (see `proxy.Request`), so a
request id can't ride the `Request` object. But capture and transport must
agree on ONE id per call: the `Recorder` writes a `request_id` into its JSONL
row, and the transport layer (`Forward`, the SDK clients) stamps
`x-request-id` on the upstream hop — downstream token recorders (the TITO
gateway) echo that header into their own per-turn records. If each layer
minted its own id, the message-level row and the token-level record for the
same call could never be joined.

A `ContextVar` is the seam: the outermost interested layer (the `Recorder`,
when present) mints the id and binds it for the duration of the handler call;
inner layers reuse a bound id and only mint their own when nothing upstream
bound one. Works unchanged across `await` within one handler invocation and
never leaks across concurrent calls.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

current_request_id: ContextVar[str | None] = ContextVar("abridge_request_id", default=None)


def mint_request_id() -> str:
    return uuid.uuid4().hex


def get_or_mint_request_id() -> str:
    """The id bound by an outer capture layer, or a fresh one."""
    bound = current_request_id.get()
    return bound if bound else mint_request_id()


__all__ = ["current_request_id", "get_or_mint_request_id", "mint_request_id"]
