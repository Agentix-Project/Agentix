"""`Recorder` — capture rollout traffic at the tunnel, one JSONL line per call.

The tunnel is the one place every LLM call an agent makes passes through,
so it is the natural recording point for rollout data collection: wrap any
handler client in `Recorder(client, path)` and hand the wrapper to
`Proxy(...)` — neither the agent nor the upstream can tell the difference.

Each served request appends one line::

    {"ts": ..., "path": "/v1/messages", "request": {...},
     "response": {"status_code": 200, "media_type": "...", "body": ...}}

A handler that raises records `{"error": ...}` instead of `"response"` and
re-raises — a failed call is signal, not something to lose. JSON bodies are
recorded as objects; anything else (e.g. a pre-rendered SSE blob) as text.

Handlers run on the event loop, so appends never interleave; each line is
flushed as it is written so the file is complete up to the last call even
if the process dies mid-rollout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Any

from .proxy import ClientResponse, Handler, Request, _collect_handlers


class Recorder:
    """Wrap a handler client; record every (request, response) pair it serves.

    Exposes the inner client's routes via `abridge_routes()` (the blessed
    dynamic-route seam), delegates `environ(...)`, and closes both the inner
    client and the record file on `aclose()` — so `Proxy.stop()` tears the
    whole stack down once, as usual.
    """

    def __init__(self, client: Any, path: str | Path) -> None:
        self._client = client
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = self._path.open("a", encoding="utf-8")

    def abridge_routes(self) -> dict[str, Handler]:
        return {path: self._recording(path, handler) for path, handler in _collect_handlers(self._client).items()}

    def _recording(self, path: str, handler: Handler) -> Handler:
        async def record(request: Request) -> ClientResponse:
            line: dict[str, Any] = {"ts": time.time(), "path": path, "request": request.body}
            try:
                response = await handler(request)
            except BaseException as exc:
                line["error"] = f"{type(exc).__name__}: {exc}"
                self._write(line)
                raise
            line["response"] = {
                "status_code": response.status_code,
                "media_type": response.media_type,
                "body": _decode_body(response),
            }
            self._write(line)
            return response

        return record

    def _write(self, line: dict[str, Any]) -> None:
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
            self._file.close()


def _decode_body(response: ClientResponse) -> Any:
    if response.media_type == "application/json":
        try:
            return json.loads(response.body)
        except (ValueError, UnicodeDecodeError):
            pass
    return response.body.decode("utf-8", "replace")


__all__ = ["Recorder"]
