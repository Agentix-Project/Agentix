"""Runtime transport wire types.

Every type here is part of the runtime wire surface between
`RuntimeClient` (orchestrator side), the runtime server (sandbox side),
and the worker subprocess. Both client and server import from here.

Wire encoding: callable identity travels as an import path
(`module::qualname`). Args, kwargs, and return values travel as stdlib
pickle blobs so arbitrary Python values can cross the boundary.
"""

from __future__ import annotations

from pydantic import BaseModel

from agentix.runtime.shared.callables import RemoteCallable
from agentix.runtime.shared.idents import CallId


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class RemoteRequest(BaseModel):
    """One remote call.

      - `callable`: a `RemoteCallable` (str subclass holding
        `module::qualname`). `.resolve()` imports the fn on the worker;
        `RemoteCallable._resolve(fn)` builds one on the host.
      - `arguments`: pickle.dumps((args, kwargs)).

    No display name on the wire — both ends compute it locally from
    their fn reference for log lines and error messages.
    """

    model_config = {"arbitrary_types_allowed": True}

    callable: RemoteCallable
    arguments: bytes
    call_id: CallId | None = None


class RemoteError(BaseModel):
    type: str
    message: str
    traceback: str | None = None
    cancelled: bool = False


class RemoteResponse(BaseModel):
    """Internal worker response. `value` is a pickle blob on success."""

    ok: bool
    value: bytes | None = None
    error: RemoteError | None = None
