"""Protocol-neutral bridge wire types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NAMESPACE = "/abridge-mitm"

ProxyEvent = dict[str, Any]
ProxyAction = dict[str, Any]
ResponseEnvelope = dict[str, Any]


@dataclass
class ProxyHandle:
    id: str
    url: str
    port: int
    forwarder_url: str
    forwarder_port: int
    mode: str
