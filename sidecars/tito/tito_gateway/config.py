"""Configuration objects for the TITO Gateway wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tito_gateway.discovery import DEFAULT_BACKEND_PROBE_CANDIDATES


_VALID_APPEND_ROLES = frozenset({"tool", "user", "system"})


@dataclass(frozen=True)
class TITOGatewayConfig:
    """Miles-compatible configuration for the standalone gateway wrapper."""

    hf_checkpoint: str
    backend_url: str | None = None
    # Explicit multi-backend pool (sglang/vLLM replicas). When set, these are
    # used as-is and single-URL discovery is skipped; `backend_url` is left as
    # the first entry for callers that read it.
    backend_urls: tuple[str, ...] = ()
    routing_policy: str = "sticky"
    chat_template_path: str | None = None
    apply_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    tito_model: str = "default"
    tito_allowed_append_roles: tuple[str, ...] = ("tool",)
    session_server_ip: str = "127.0.0.1"
    session_server_port: int = 30000
    miles_router_timeout: float = 600.0
    backend_probe_candidates: tuple[str, ...] = field(default_factory=lambda: DEFAULT_BACKEND_PROBE_CANDIDATES)
    backend_probe_timeout: float = 0.25

    def __post_init__(self) -> None:
        if not self.hf_checkpoint:
            raise ValueError("hf_checkpoint is required for TITO token tracking")

        normalized_roles = tuple(dict.fromkeys(role.lower() for role in self.tito_allowed_append_roles))
        invalid = sorted(set(normalized_roles) - _VALID_APPEND_ROLES)
        if invalid:
            raise ValueError(f"unsupported tito append roles: {invalid}")
        object.__setattr__(self, "tito_allowed_append_roles", normalized_roles or ("tool",))

        if self.routing_policy not in ("sticky", "round_robin"):
            raise ValueError(
                f"routing_policy must be 'sticky' or 'round_robin'; got {self.routing_policy!r}"
            )

    @classmethod
    def from_cli_values(
        cls,
        *,
        hf_checkpoint: str,
        backend_url: str | None,
        chat_template_path: str | None,
        apply_chat_template_kwargs: str | None,
        tito_model: str,
        tito_allowed_append_roles: list[str],
        session_server_ip: str,
        session_server_port: int,
        miles_router_timeout: float,
        backend_probe_candidates: list[str] | None = None,
        backend_probe_timeout: float = 0.25,
    ) -> "TITOGatewayConfig":
        kwargs: dict[str, Any] = {}
        if apply_chat_template_kwargs:
            parsed = json.loads(apply_chat_template_kwargs)
            if not isinstance(parsed, dict):
                raise ValueError("--apply-chat-template-kwargs must decode to a JSON object")
            kwargs = parsed

        return cls(
            hf_checkpoint=hf_checkpoint,
            backend_url=backend_url,
            chat_template_path=chat_template_path,
            apply_chat_template_kwargs=kwargs,
            tito_model=tito_model,
            tito_allowed_append_roles=tuple(tito_allowed_append_roles),
            session_server_ip=session_server_ip,
            session_server_port=session_server_port,
            miles_router_timeout=miles_router_timeout,
            backend_probe_candidates=tuple(backend_probe_candidates or DEFAULT_BACKEND_PROBE_CANDIDATES),
            backend_probe_timeout=backend_probe_timeout,
        )

    def as_miles_namespace(self):
        """Return an argparse-like namespace for vendored Miles session code."""
        from types import SimpleNamespace

        return SimpleNamespace(
            hf_checkpoint=self.hf_checkpoint,
            chat_template_path=self.chat_template_path,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            tito_model=self.tito_model,
            tito_allowed_append_roles=list(self.tito_allowed_append_roles),
            session_server_ip=self.session_server_ip,
            session_server_port=self.session_server_port,
            miles_router_timeout=self.miles_router_timeout,
        )
