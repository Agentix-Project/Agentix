"""Configuration objects for the TITO Gateway wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field

from .discovery import DEFAULT_BACKEND_PROBE_CANDIDATES

_VALID_APPEND_ROLES = frozenset({"tool", "user", "system"})


@dataclass(frozen=True)
class TITOGatewayConfig:
    """Configuration for the standalone TITO gateway wrapper."""

    hf_checkpoint: str
    backend_url: str | None = None
    # Explicit multi-backend pool — replicas of ONE backend kind. When set,
    # these are used as-is and single-URL discovery is skipped; `backend_url`
    # is left as the first entry for callers that read it.
    backend_urls: tuple[str, ...] = ()
    # Which token-exact dialect the backend speaks: "sglang" (input_ids +
    # meta_info extensions) or "vllm" (>= 0.24.0, render/generate/derender).
    backend_kind: str = "sglang"
    routing_policy: str = "sticky"
    # Execute Python shipped inside the checkpoint repo when loading the
    # tokenizer. Off by default; opt in only for checkpoints you trust.
    trust_remote_code: bool = False
    chat_template_path: str | None = None
    tito_model: str = "default"
    tito_allowed_append_roles: tuple[str, ...] = ("tool",)
    session_server_ip: str = "127.0.0.1"
    session_server_port: int = 30000
    router_timeout: float = 600.0
    backend_probe_candidates: tuple[str, ...] = field(default_factory=lambda: DEFAULT_BACKEND_PROBE_CANDIDATES)
    backend_probe_timeout: float = 0.25
    # Durable capture: when set, every committed turn appends one
    # `tito.record.v1` line to `<record_dir>/<session_id>.jsonl` (flushed per
    # turn). Unset = in-memory trajectories only (unchanged default).
    record_dir: str | None = None
    # Long-running-gateway lifecycle (both optional, default off): evict
    # sessions idle beyond the TTL, and LRU-evict beyond max_sessions.
    # Eviction flushes+finalizes the session's record file first and never
    # touches a session with in-flight requests.
    session_ttl_seconds: float | None = None
    max_sessions: int | None = None

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

        if self.backend_kind not in ("sglang", "vllm"):
            raise ValueError(f"backend_kind must be 'sglang' or 'vllm'; got {self.backend_kind!r}")

        if self.session_ttl_seconds is not None and self.session_ttl_seconds <= 0:
            raise ValueError(f"session_ttl_seconds must be > 0; got {self.session_ttl_seconds!r}")
        if self.max_sessions is not None and self.max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1; got {self.max_sessions!r}")

    @classmethod
    def from_cli_values(
        cls,
        *,
        hf_checkpoint: str,
        backend_url: str | None,
        chat_template_path: str | None,
        tito_model: str,
        tito_allowed_append_roles: list[str],
        session_server_ip: str,
        session_server_port: int,
        router_timeout: float,
        backend_urls: list[str] | None = None,
        backend_kind: str = "sglang",
        routing_policy: str = "sticky",
        trust_remote_code: bool = False,
        backend_probe_candidates: list[str] | None = None,
        backend_probe_timeout: float = 0.25,
        record_dir: str | None = None,
        session_ttl_seconds: float | None = None,
        max_sessions: int | None = None,
    ) -> TITOGatewayConfig:
        return cls(
            hf_checkpoint=hf_checkpoint,
            backend_url=backend_url,
            backend_urls=tuple(backend_urls or ()),
            backend_kind=backend_kind,
            routing_policy=routing_policy,
            trust_remote_code=trust_remote_code,
            chat_template_path=chat_template_path,
            tito_model=tito_model,
            tito_allowed_append_roles=tuple(tito_allowed_append_roles),
            session_server_ip=session_server_ip,
            session_server_port=session_server_port,
            router_timeout=router_timeout,
            backend_probe_candidates=tuple(backend_probe_candidates or DEFAULT_BACKEND_PROBE_CANDIDATES),
            backend_probe_timeout=backend_probe_timeout,
            record_dir=record_dir,
            session_ttl_seconds=session_ttl_seconds,
            max_sessions=max_sessions,
        )

    def as_session_args(self):
        """Return an argparse-like namespace consumed by the engine session routes."""
        from types import SimpleNamespace

        return SimpleNamespace(
            hf_checkpoint=self.hf_checkpoint,
            backend_kind=self.backend_kind,
            chat_template_path=self.chat_template_path,
            tito_model=self.tito_model,
            tito_allowed_append_roles=list(self.tito_allowed_append_roles),
            trust_remote_code=self.trust_remote_code,
            session_server_ip=self.session_server_ip,
            session_server_port=self.session_server_port,
            router_timeout=self.router_timeout,
            record_dir=self.record_dir,
            session_ttl_seconds=self.session_ttl_seconds,
            max_sessions=self.max_sessions,
        )
