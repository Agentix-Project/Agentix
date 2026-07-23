"""`Recorder` — capture rollout traffic at the tunnel, one JSONL line per call.

The tunnel is the one place every LLM call an agent makes passes through,
so it is the natural recording point for rollout data collection: wrap any
handler client in `Recorder(client, path)` and hand the wrapper to
`Proxy(...)` — neither the agent nor the upstream can tell the difference.

Each served request appends one line::

    {"ts": ..., "path": "/v1/messages", "request_id": "<32 hex>",
     "session_id": ...,                      # only when the Recorder has one
     "request": {...},
     "response": {"status_code": 200, "media_type": "...", "body": ...}}

`request_id` is minted per call and bound on the `current_request_id`
context var for the duration of the handler, so the transport layer
(`Forward` / the SDK clients) stamps the SAME id as `x-request-id` on the
upstream hop — a downstream token recorder's per-turn record and this row
join on it. `session_id`, when given, identifies the rollout the wrapped
client serves (pass the same value as the client's session identity).
Without these keys, rows from a retried call (e.g. an agent retry after a
tunnel 504 produced an orphan success row) are only deduplicable by
request-body equality.

A handler that raises records `{"error": ...}` instead of `"response"` and
re-raises — a failed call is signal, not something to lose. JSON bodies are
recorded as objects; anything else (e.g. a pre-rendered SSE blob) as text.

Handlers run on the event loop, so appends never interleave; each line is
flushed as it is written so the file is complete up to the last call even
if the process dies mid-rollout. The file opens lazily on the first record,
so a Recorder that never serves (e.g. a route-enumeration probe) leaves no
empty file behind.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Any

from ._request_id import current_request_id, mint_request_id
from .proxy import ClientResponse, Handler, Request, _collect_handlers


class Recorder:
    """Wrap a handler client; record every (request, response) pair it serves.

    Exposes the inner client's routes via `abridge_routes()` (the blessed
    dynamic-route seam), delegates `environ(...)`, and closes both the inner
    client and the record file on `aclose()` — so `Proxy.stop()` tears the
    whole stack down once, as usual.
    """

    def __init__(self, client: Any, path: str | Path, *, session_id: str | None = None) -> None:
        self._client = client
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id
        self._file: IO[str] | None = None

    def abridge_routes(self) -> dict[str, Handler]:
        return {path: self._recording(path, handler) for path, handler in _collect_handlers(self._client).items()}

    def _recording(self, path: str, handler: Handler) -> Handler:
        async def record(request: Request) -> ClientResponse:
            # Reuse an id bound by an even-outer layer; otherwise mint here.
            # Binding it makes the transport's upstream `x-request-id` equal
            # this row's `request_id`.
            request_id = current_request_id.get() or mint_request_id()
            line: dict[str, Any] = {"ts": time.time(), "path": path, "request_id": request_id}
            if self._session_id is not None:
                line["session_id"] = self._session_id
            line["request"] = request.body
            token = current_request_id.set(request_id)
            try:
                response = await handler(request)
            except BaseException as exc:
                line["error"] = f"{type(exc).__name__}: {exc}"
                self._write(line)
                raise
            finally:
                current_request_id.reset(token)
            line["response"] = {
                "status_code": response.status_code,
                "media_type": response.media_type,
                "body": _decode_body(response),
            }
            self._write(line)
            return response

        return record

    def _write(self, line: dict[str, Any]) -> None:
        if self._file is None or self._file.closed:
            self._file = self._path.open("a", encoding="utf-8")
        self._file.write(json.dumps(line, ensure_ascii=False, default=repr) + "\n")
        self._file.flush()

    def environ(self, handle: Any) -> dict[str, str]:
        return self._client.environ(handle)

    async def aclose(self) -> None:
        try:
            aclose = getattr(self._client, "aclose", None)
            if aclose is not None:
                await aclose()
        finally:
            if self._file is not None:
                self._file.close()


def _decode_body(response: ClientResponse) -> Any:
    if response.media_type == "application/json":
        try:
            return json.loads(response.body)
        except (ValueError, UnicodeDecodeError):
            pass
    return response.body.decode("utf-8", "replace")


__all__ = ["Recorder"]
