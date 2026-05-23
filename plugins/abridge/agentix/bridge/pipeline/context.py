"""HTTP request context for the interceptor pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestContext:
    flow: Any
    host: str
    path: str
    method: str
    body: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def from_http_flow(cls, flow: Any) -> RequestContext:
        return cls(
            flow=flow,
            host=flow.request.pretty_host,
            path=flow.request.path,
            method=flow.request.method.upper(),
        )

    @property
    def path_without_query(self) -> str:
        return self.path.split("?", 1)[0]

    def parsed_body(self) -> dict[str, Any]:
        if self.body is None:
            from agentix.bridge.events import json_request_body

            self.body = json_request_body(self.flow)
        return self.body
