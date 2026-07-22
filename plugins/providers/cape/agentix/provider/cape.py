"""CAPE provider: sandbox CRUD via the `cape` CLI on a lease-based GPU pool.

CAPE is a lease-based GPU capacity pool: a submitter-side `cape` CLI
sends workload requests to a controller, which schedules them onto
pool nodes with strong per-request isolation. Requests that share a
`--session-key` land in the same warm session — the session alternates
between RUNNING_COMMAND and CACHED_IDLE (a *serial* model: one command
at a time per session), so a session's writable state survives across
commands until the session is reclaimed.

Mapping to Agentix: one sandbox is exactly one long-lived CAPE request
inside a per-sandbox session (`agentix-cape-...` session keys). The
provider never submits a second request into the same session, so it
does not depend on same-session concurrency or on cross-request
workspace persistence.

  - `create()` submits a runtime request whose workload prints an
    `AGENTIX_ENDPOINT <host> <port>` marker line to stdout (first IP of
    `hostname -i`, plus the assigned bind port) and then execs the
    bundle's `/nix/runtime/bootstrap.sh`. The provider polls
    `cape status` (to detect early death) and `cape logs` (to find the
    marker), then health-checks `GET /health` on the discovered
    endpoint with a raw TCP probe (never an env-proxy-aware HTTP
    client, which would hang behind a corp proxy or SSH tunnel).
  - `delete()` cancels the runtime request; bookkeeping is dropped only
    after the controller confirms the cancel, so a failed cancel can be
    retried with another `delete()`.
  - `config.bundle` for this backend is an OPAQUE node-visible path to
    an already-extracted bundle tree; it is bind-mounted read-only at
    `/nix`. How the bundle tree gets onto CAPE nodes (shared FS, prior
    upload) is deliberately out of scope — there is no `BundleDeployer`
    in this iteration.

ASSUMED CLI CONTRACT — the `cape run/status/logs/cancel` verb surface
implemented in `_CapeCli` was reverse-documented from a sibling
project's adapter of the same CLI and has only ever been
exercised against fakes and emulators; the official CAPE CLI has never
been obtained. Endpoint discovery additionally assumes `cape logs` can
return the stdout of a still-RUNNING request. Verify every verb, flag,
and the `cape status` JSON schema against the real CLI before
production use (see the provider README's "Contract status" checklist).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field

from agentix.provider.base import (
    Sandbox,
    SandboxConfig,
    SandboxId,
    SandboxInfo,
    SandboxProvider,
    SandboxResource,
)
from agentix.runtime import BIND_PORT_ENV, BUNDLE_NIX_ROOT, BUNDLE_RUNTIME_ENTRYPOINT

logger = logging.getLogger("agentix.provider.cape")

_REQUEST_ID_RE = re.compile(r"req-[A-Za-z0-9][A-Za-z0-9_.-]*")
"""Shape of the request id `cape run` prints on stdout (assumed contract)."""

_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "LOST", "INFEASIBLE"})
"""Terminal request states in the `cape status` JSON (assumed contract)."""

_REASON_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")
"""Characters stripped from `cape cancel --reason` strings."""

_ENDPOINT_MARKER = "AGENTIX_ENDPOINT"
"""Prefix of the stdout marker line the boot script prints before exec."""

_MEMORY_RE = re.compile(r"(\d+)\s*([kmgt])?i?b?", re.IGNORECASE)
_MEMORY_UNIT_BYTES = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}

_REAP_TIMEOUT_SECONDS = 5.0
"""Bounded wait for a killed `cape` subprocess to be reaped."""


class CapeProviderConfig(BaseModel):
    """Submitter-side settings for the CAPE backend.

    Every field is optional or defaulted so `CapeProvider()` constructs
    with zero arguments — the plugin registry instantiates providers
    with `cls()`. Anything unset falls back to environment variables at
    first use.
    """

    binary: str | None = Field(
        default=None,
        description="`cape` CLI to invoke. Resolution order: this field, then the "
        "`CAPE_BINARY` env var, then bare `cape` on PATH. A value containing a "
        "path separator must point at an existing executable file.",
    )
    controller_url: str | None = Field(
        default=None,
        description="CAPE controller URL. Falls back to `CAPE_CONTROLLER_URL`; "
        "required at first use.",
    )
    token_file: str | None = Field(
        default=None,
        description="File holding the CAPE token (read per operation, so rotation "
        "works). Falls back to `CAPE_TOKEN_FILE`. The file must not be "
        "group/other accessible (mode & 0o077 == 0).",
    )
    token_env: str = Field(
        default="CAPE_TOKEN",
        description="Env var consulted for the token when no token file is configured.",
    )
    pool: str | None = Field(default=None, description="Optional CAPE pool name.")
    user: str | None = Field(default=None, description="Optional CAPE user identity.")
    workspace_root: str = Field(
        default="/workspace",
        description="Node-side base directory; each sandbox uses "
        "`<workspace_root>/<sandbox_id>` as its session workspace and cwd.",
    )
    cpu_cores: int | None = Field(
        default=None,
        description="Default `--cpu-cores` when `SandboxConfig.resource.cpu` is unset.",
    )
    memory_gb: int | None = Field(
        default=None,
        description="Default `--memory-gb` when `SandboxConfig.resource.memory` is unset.",
    )
    max_duration_seconds: int = Field(
        default=14400,
        description="`--max-duration-seconds` for the long-lived runtime request.",
    )
    isolation_policy: str = Field(default="default", description="`--isolation-policy` value.")
    runtime_adapter: str | None = Field(
        default=None,
        description="Optional `--runtime-adapter` (e.g. `apptainer`).",
    )
    extra_binds: list[str] = Field(
        default_factory=list,
        description="Raw `src:dst[:ro|rw]` bind specs passed through in addition to "
        "the bundle's `/nix` bind.",
    )
    runtime_port_base: int = Field(
        default=8710,
        description="First in-sandbox runtime port (`AGENTIX_BIND_PORT`). Each live "
        "sandbox of this provider gets a distinct port from "
        "`[runtime_port_base, runtime_port_base + runtime_port_span)` so "
        "same-node / same-tunnel sandboxes cannot answer each other's probes. "
        "Collisions with other submitters on the same node cannot be reserved "
        "from this side — see the README's known limitations.",
    )
    runtime_port_span: int = Field(
        default=200,
        description="Size of the per-provider runtime port range; bounds the number "
        "of concurrently live sandboxes per provider instance.",
    )
    url_template: str = Field(
        default="http://{host}:{port}",
        description="How to build `runtime_url` from the discovered endpoint. Users "
        "reaching CAPE nodes through an SSH tunnel can set e.g. "
        "`http://127.0.0.1:{port}`.",
    )
    client_timeout_seconds: float = Field(
        default=60.0,
        description="Per-`cape`-subprocess timeout.",
    )
    create_timeout_seconds: float = Field(
        default=600.0,
        description="Total budget for endpoint discovery plus the health probe "
        "(lease queuing can be slow).",
    )
    poll_interval_seconds: float = Field(
        default=2.0,
        description="Sleep between endpoint-discovery attempts.",
    )
    transient_failure_limit: int = Field(
        default=5,
        description="Consecutive `cape status`/`cape logs` failures tolerated during "
        "endpoint discovery before the create is failed (a single controller "
        "blip must not cancel a lease that already queued onto a GPU).",
    )


def _resolve_binary(config: CapeProviderConfig) -> str:
    """Resolve the `cape` binary: config field > `CAPE_BINARY` env > PATH.

    Synchronous file IO — call via `asyncio.to_thread` from async code.
    """
    value = config.binary or os.environ.get("CAPE_BINARY") or "cape"
    has_separator = os.sep in value or (os.altsep is not None and os.altsep in value)
    if has_separator:
        path = Path(value).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(
                f"cape binary {value!r} (from CapeProviderConfig.binary or CAPE_BINARY) "
                f"must be an existing executable file"
            )
        return str(path)
    return value


def _resolve_controller_url(config: CapeProviderConfig) -> str:
    url = config.controller_url or os.environ.get("CAPE_CONTROLLER_URL")
    if not url:
        raise RuntimeError(
            "CAPE controller URL is not configured: set CapeProviderConfig.controller_url "
            "or the CAPE_CONTROLLER_URL environment variable"
        )
    return url


def _read_token(config: CapeProviderConfig) -> str:
    """Read the CAPE token, fresh on every operation (supports rotation).

    The token file (config or `CAPE_TOKEN_FILE`) wins over the token env
    var. Empty and whitespace-containing tokens are rejected with an
    error naming the offending source. The literal token value must
    never reach logs or exception text — see `_redact`.

    Synchronous file IO — call via `asyncio.to_thread` from async code
    (the token file often lives on NFS, which must not block the loop).
    """
    token_file = config.token_file or os.environ.get("CAPE_TOKEN_FILE")
    if token_file:
        path = Path(token_file).expanduser()
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise RuntimeError(f"CAPE token file {path} is not readable: {exc}") from exc
        if mode & 0o077 != 0:
            raise RuntimeError(
                f"CAPE token file {path} must not be group/other accessible; run `chmod 600` on it"
            )
        token = path.read_text(encoding="utf-8").strip()
        source = f"CAPE token file {path}"
    else:
        token = os.environ.get(config.token_env, "")
        source = f"CAPE token env var {config.token_env}"
    if not token:
        raise RuntimeError(f"{source} is empty; provide a CAPE token")
    if any(ch.isspace() for ch in token):
        raise RuntimeError(f"{source} contains whitespace; refusing to use it")
    return token


def _redact(text: str, token: str) -> str:
    """Replace the literal token value with `<redacted>` in `text`."""
    if token and token in text:
        return text.replace(token, "<redacted>")
    return text


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    """Done-callback that retrieves a detached task's exception so it is
    never reported as `Task exception was never retrieved`."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("detached cape task failed: %r", exc)


def _cpu_cores(resource: SandboxResource | None, config: CapeProviderConfig) -> int | None:
    if resource is not None and resource.cpu is not None:
        # CAPE cores are whole integers; round a fractional request up so
        # the granted capacity always covers what was asked for.
        return max(1, math.ceil(resource.cpu))
    return config.cpu_cores


def _memory_gb(resource: SandboxResource | None, config: CapeProviderConfig) -> int | None:
    if resource is None or resource.memory is None:
        return config.memory_gb
    memory = resource.memory
    if isinstance(memory, int):
        num_bytes = memory
    else:
        match = _MEMORY_RE.fullmatch(memory.strip())
        if match is None:
            raise RuntimeError(
                f"cannot map memory {memory!r} to --memory-gb; use container CLI "
                f"unit syntax, e.g. `16g`"
            )
        unit = (match.group(2) or "").lower()
        num_bytes = int(match.group(1)) * (_MEMORY_UNIT_BYTES[unit] if unit else 1)
    return max(1, math.ceil(num_bytes / (1 << 30)))


def _boot_script() -> str:
    """Workload for the long-lived runtime request.

    Prints the `AGENTIX_ENDPOINT <host> <port>` marker to stdout (first
    IP of `hostname -i`, plus the assigned bind port) and execs the
    bundle's bootstrap entry point. The marker is read back host-side
    via `cape logs` during endpoint discovery — nothing is written to
    the workspace, so discovery does not depend on cross-request
    workspace persistence or on a second same-session command.
    """
    return (
        f'printf "{_ENDPOINT_MARKER} %s %s\\n" "$(hostname -i | cut -d" " -f1)" '
        f'"${{{BIND_PORT_ENV}}}" && '
        f"exec {BUNDLE_RUNTIME_ENTRYPOINT}"
    )


def _parse_endpoint(text: str) -> tuple[str, int] | None:
    """Find the `AGENTIX_ENDPOINT <host> <port>` marker line in workload stdout.

    The explicit marker prefix keeps unrelated two-token output (banner
    lines like `GPU 0`) from being misread as an endpoint.
    """
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == _ENDPOINT_MARKER and parts[2].isdigit():
            return parts[1], int(parts[2])
    return None


class _CapeCli:
    """The ONE place that knows the assumed `cape` CLI verb surface.

    Every method's contract (verbs, flags, stdout/JSON shapes, terminal
    states) is an assumption reverse-documented from a sibling
    project's adapter — it has never been checked against a real
    `cape` binary.
    Keep all CLI knowledge in this class so a contract correction after
    real-CLI verification is a single-class change.
    """

    def __init__(self, config: CapeProviderConfig) -> None:
        self._config = config

    def _common(self, token: str) -> list[str]:
        return ["--controller-url", _resolve_controller_url(self._config), "--token", token]

    async def _exec(self, argv: Sequence[str], *, token: str, verb: str) -> tuple[int, str, str]:
        """Run one `cape` subprocess; stdout/stderr come back token-redacted.

        The subprocess is never abandoned: local timeout and external
        cancellation both kill it and reap it within a bounded window
        (a leaked `cape run` could otherwise still submit work).
        """
        binary = await asyncio.to_thread(_resolve_binary, self._config)
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"cape binary {binary!r} not found; set CapeProviderConfig.binary or "
                f"CAPE_BINARY, or install `cape` on PATH"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._config.client_timeout_seconds
            )
        except asyncio.CancelledError:
            proc.kill()
            # Bounded reap via `proc.wait()` (not `communicate()`): a killed
            # child cannot block `wait()` on a full pipe, and a grandchild
            # holding the pipe write-end open cannot stall us past the bound.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_SECONDS)
            raise
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_SECONDS)
            raise RuntimeError(
                f"cape {verb} timed out after {self._config.client_timeout_seconds}s"
            ) from None
        return (
            proc.returncode or 0,
            _redact(stdout.decode(errors="replace"), token),
            _redact(stderr.decode(errors="replace"), token),
        )

    async def run(
        self,
        *,
        session_key: str,
        workspace: str,
        cwd: str,
        image: str,
        gpus: int,
        workload: Sequence[str],
        binds: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cpu_cores: int | None = None,
        memory_gb: int | None = None,
    ) -> str:
        """ASSUMED CLI contract — verify against the real `cape` CLI before production use.

        Submits one workload request and returns its request id:

            cape run --controller-url U --token T [--pool P] [--user USR]
                --workspace WS --image IMG --gpus N [--cpu-cores N]
                [--memory-gb N] --gpu-mode whole --isolation-policy X
                [--runtime-adapter Y] --max-duration-seconds T --cwd CWD
                --session-key KEY [--bind src:dst[:ro|rw]]... [--env K=V]...
                -- <workload argv>

        stdout, after stripping blank lines, must be exactly one line
        matching `req-[A-Za-z0-9][A-Za-z0-9_.-]*`.
        """
        token = await asyncio.to_thread(_read_token, self._config)
        cfg = self._config
        argv: list[str] = ["run", *self._common(token)]
        if cfg.pool:
            argv += ["--pool", cfg.pool]
        if cfg.user:
            argv += ["--user", cfg.user]
        argv += ["--workspace", workspace, "--image", image, "--gpus", str(int(gpus))]
        if cpu_cores is not None:
            argv += ["--cpu-cores", str(int(cpu_cores))]
        if memory_gb is not None:
            argv += ["--memory-gb", str(int(memory_gb))]
        argv += ["--gpu-mode", "whole", "--isolation-policy", cfg.isolation_policy]
        if cfg.runtime_adapter:
            argv += ["--runtime-adapter", cfg.runtime_adapter]
        argv += ["--max-duration-seconds", str(int(cfg.max_duration_seconds))]
        argv += ["--cwd", cwd, "--session-key", session_key]
        for bind in binds:
            argv += ["--bind", bind]
        for key, value in (env or {}).items():
            argv += ["--env", f"{key}={value}"]
        argv += ["--", *workload]
        rc, stdout, stderr = await self._exec(argv, token=token, verb="run")
        if rc != 0:
            raise RuntimeError(f"cape run failed (rc={rc}): {stderr}")
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if len(lines) != 1 or not _REQUEST_ID_RE.fullmatch(lines[0]):
            raise RuntimeError(
                f"cape run did not print exactly one request id "
                f"(stdout={stdout!r}, stderr={stderr})"
            )
        return lines[0]

    async def status(self, request_id: str) -> dict[str, object]:
        """ASSUMED CLI contract — verify against the real `cape` CLI before production use.

        `cape status <req> --controller-url U --token T` prints one JSON
        object with at least a `state` field; terminal states are
        COMPLETED / FAILED / CANCELLED / EXPIRED / LOST / INFEASIBLE. A
        `request_id` field, when present and non-null, must echo the
        queried id — a mismatch means the CLI answered for a different
        request and is treated as an infrastructure error, not trusted.
        """
        token = await asyncio.to_thread(_read_token, self._config)
        rc, stdout, stderr = await self._exec(
            ["status", request_id, *self._common(token)], token=token, verb="status"
        )
        if rc != 0:
            raise RuntimeError(f"cape status {request_id} failed (rc={rc}): {stderr}")
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"cape status {request_id} printed non-JSON output: {stdout!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"cape status {request_id} printed JSON of type "
                f"{type(payload).__name__}, expected an object"
            )
        echoed = payload.get("request_id")
        if echoed is not None and echoed != request_id:
            raise RuntimeError(
                f"cape status {request_id} returned status for a different request "
                f"({echoed!r}); refusing to trust it"
            )
        return payload

    async def cancel(self, request_id: str, reason: str) -> bool:
        """ASSUMED CLI contract — verify against the real `cape` CLI before production use.

        `cape cancel <req> --controller-url U --token T --reason R`.
        Returns True only when the CLI confirmed the cancel (rc=0);
        ordinary failures are swallowed (best-effort) after a redacted
        warning. External cancellation is NOT swallowed: the in-flight
        cancel RPC gets a bounded shielded window to reach the
        controller, then CancelledError is re-raised so callers such as
        `asyncio.timeout` / TaskGroup teardown still observe it.
        """
        safe_reason = _REASON_UNSAFE_RE.sub("_", reason)
        task = asyncio.ensure_future(self._cancel_once(request_id, safe_reason))
        task.add_done_callback(_consume_task_result)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=self._config.client_timeout_seconds + _REAP_TIMEOUT_SECONDS,
                )
            raise
        except Exception:
            logger.warning(
                "cape cancel %s (reason=%s) failed; ignoring",
                request_id,
                safe_reason,
                exc_info=True,
            )
            return False

    async def _cancel_once(self, request_id: str, reason: str) -> bool:
        token = await asyncio.to_thread(_read_token, self._config)
        rc, _stdout, stderr = await self._exec(
            ["cancel", request_id, *self._common(token), "--reason", reason],
            token=token,
            verb="cancel",
        )
        if rc != 0:
            logger.warning(
                "cape cancel %s (reason=%s) exited rc=%d: %s",
                request_id,
                reason,
                rc,
                stderr.strip(),
            )
            return False
        return True

    async def logs(self, request_id: str, *, check: bool = False) -> tuple[str, str]:
        """ASSUMED CLI contract — verify against the real `cape` CLI before production use.

        `cape logs <req> --controller-url U --token T` prints the
        workload's stdout/stderr. ADDITIONAL ASSUMPTION introduced by
        this provider: `cape logs` can return the stdout of a
        still-RUNNING request — endpoint discovery depends on it, and
        the reference adapter only ever called logs on terminal
        requests. Verify this explicitly against the real CLI.

        With `check=False` (default) this is best-effort — failures
        collapse to `("", "")` (error-message enrichment only). With
        `check=True` a failed invocation raises RuntimeError so endpoint
        discovery can tell a broken verb apart from a marker that simply
        has not been printed yet.
        """
        try:
            token = await asyncio.to_thread(_read_token, self._config)
            rc, stdout, stderr = await self._exec(
                ["logs", request_id, *self._common(token)], token=token, verb="logs"
            )
        except Exception:
            if check:
                raise
            return "", ""
        if rc != 0:
            if check:
                raise RuntimeError(f"cape logs {request_id} failed (rc={rc}): {stderr}")
            return "", ""
        return stdout, stderr


@dataclass
class _CapeSandboxRecord:
    """Bookkeeping for one live sandbox."""

    request_id: str
    session_key: str
    workspace: str
    runtime_url: str
    runtime_port: int


class CapeProvider(SandboxProvider):
    """Sandbox CRUD via the `cape` CLI (assumed contract; see module docstring)."""

    def __init__(self, config: CapeProviderConfig | None = None) -> None:
        self.config = config or CapeProviderConfig()
        self._cli = _CapeCli(self.config)
        self._sandboxes: dict[SandboxId, _CapeSandboxRecord] = {}
        # Runtime ports handed out to live/in-flight sandboxes, so two
        # sandboxes of this provider can never share an AGENTIX_BIND_PORT
        # (same-node or same-SSH-tunnel siblings would otherwise answer
        # each other's health probes). Released on delete / failed create.
        self._inflight_ports: set[int] = set()

    def _allocate_port(self) -> int:
        base = self.config.runtime_port_base
        for port in range(base, base + self.config.runtime_port_span):
            if port not in self._inflight_ports:
                self._inflight_ports.add(port)
                return port
        raise RuntimeError(
            f"no free runtime port in [{base}, {base + self.config.runtime_port_span}); "
            f"raise runtime_port_span or delete finished sandboxes"
        )

    async def create(self, config: SandboxConfig) -> Sandbox:
        sandbox_id = SandboxId(f"cape-{uuid4().hex[:12]}")
        session_key = f"agentix-{sandbox_id}"
        workspace = f"{self.config.workspace_root.rstrip('/')}/{sandbox_id}"
        port = self._allocate_port()

        binds = [f"{config.bundle}:{BUNDLE_NIX_ROOT}:ro", *self.config.extra_binds]
        env = {BIND_PORT_ENV: str(port), **(config.env or {})}
        resource = config.resource
        gpus = resource.gpu if resource is not None and resource.gpu is not None else 0

        # The submit runs shielded: if we are cancelled mid-`cape run`, the
        # CLI still finishes inside its own timeout, the request id is
        # harvested within a bounded window, and the request is best-effort
        # cancelled. The residual window where the CLI dies after the
        # controller accepted the request cannot be closed from this side —
        # see the README's operational notes (session keys are greppable:
        # `agentix-cape-...`).
        run_task = asyncio.ensure_future(
            self._cli.run(
                session_key=session_key,
                workspace=workspace,
                cwd=workspace,
                image=config.image,
                gpus=gpus,
                workload=["sh", "-c", _boot_script()],
                binds=binds,
                env=env,
                cpu_cores=_cpu_cores(resource, self.config),
                memory_gb=_memory_gb(resource, self.config),
            )
        )
        run_task.add_done_callback(_consume_task_result)
        try:
            server_req = await asyncio.shield(run_task)
        except asyncio.CancelledError:
            self._inflight_ports.discard(port)
            harvested: str | None = None
            with contextlib.suppress(BaseException):
                harvested = await asyncio.wait_for(
                    asyncio.shield(run_task),
                    timeout=self.config.client_timeout_seconds + _REAP_TIMEOUT_SECONDS,
                )
            if harvested is not None:
                await self._cli.cancel(harvested, "agentix_create_cancelled")
            raise
        except BaseException:
            self._inflight_ports.discard(port)
            raise
        logger.info("CAPE runtime request %s submitted for sandbox %s", server_req, sandbox_id)

        # The runtime request has started: from here every failure or
        # cancellation must best-effort cancel it, drop bookkeeping, and
        # re-raise — `session()` cannot clean up a sandbox it never got.
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.config.create_timeout_seconds
            host, marker_port = await self._discover_endpoint(server_req, deadline=deadline)
            runtime_url = self.config.url_template.format(host=host, port=marker_port)
            await self._wait_healthy(server_req, runtime_url, deadline)
        except BaseException:
            self._sandboxes.pop(sandbox_id, None)
            self._inflight_ports.discard(port)
            await self._cli.cancel(server_req, "agentix_create_failed")
            raise

        self._sandboxes[sandbox_id] = _CapeSandboxRecord(
            request_id=server_req,
            session_key=session_key,
            workspace=workspace,
            runtime_url=runtime_url,
            runtime_port=port,
        )
        logger.info("Created sandbox %s at %s (request %s)", sandbox_id, runtime_url, server_req)
        # No RuntimeClient is instantiated here — `session()` stamps
        # `call_deadline` on the returned handle post-create.
        return Sandbox(sandbox_id=sandbox_id, runtime_url=runtime_url, status="running")

    async def _discover_endpoint(self, server_req: str, *, deadline: float) -> tuple[str, int]:
        """Find the node/port the runtime bound, from the workload's stdout.

        Polls `cape status` (a terminal state means the runtime died —
        fail fast with its logs) and `cape logs` (looking for the
        `AGENTIX_ENDPOINT` marker). Transient status/logs failures are
        tolerated up to `transient_failure_limit` consecutive times so a
        single controller blip cannot kill a create whose lease already
        queued onto a GPU. No second request is ever submitted into the
        session — the assumed session model is serial (RUNNING_COMMAND ↔
        CACHED_IDLE), so a same-session probe could queue forever behind
        the never-ending runtime command.
        """
        loop = asyncio.get_running_loop()
        limit = self.config.transient_failure_limit
        failures = 0
        while True:
            try:
                status = await self._cli.status(server_req)
            except RuntimeError as exc:
                failures += 1
                if failures >= limit:
                    raise RuntimeError(
                        f"cape status {server_req} failed {failures} consecutive times "
                        f"during endpoint discovery: {exc}"
                    ) from exc
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"CAPE runtime request {server_req} did not publish its endpoint "
                        f"within {self.config.create_timeout_seconds}s"
                    ) from exc
                await asyncio.sleep(self.config.poll_interval_seconds)
                continue
            state = str(status.get("state", ""))
            if state in _TERMINAL_STATES:
                stdout, stderr = await self._cli.logs(server_req)
                raise RuntimeError(
                    f"CAPE runtime request {server_req} reached terminal state {state} "
                    f"before publishing its endpoint.\n"
                    f"--- workload stdout ---\n{stdout}\n"
                    f"--- workload stderr ---\n{stderr}"
                )
            try:
                stdout, _stderr = await self._cli.logs(server_req, check=True)
            except (RuntimeError, OSError) as exc:
                failures += 1
                if failures >= limit:
                    raise RuntimeError(
                        f"cape logs {server_req} failed {failures} consecutive times "
                        f"during endpoint discovery: {exc}"
                    ) from exc
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"CAPE runtime request {server_req} did not publish its endpoint "
                        f"within {self.config.create_timeout_seconds}s"
                    ) from exc
                await asyncio.sleep(self.config.poll_interval_seconds)
                continue
            failures = 0
            endpoint = _parse_endpoint(stdout)
            if endpoint is not None:
                return endpoint
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"CAPE runtime request {server_req} did not publish its endpoint "
                    f"within {self.config.create_timeout_seconds}s"
                )
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _wait_healthy(self, server_req: str, runtime_url: str, deadline: float) -> None:
        """Probe `GET /health` until 200 or the create budget is exhausted.

        Raw TCP + a minimal hand-written HTTP request, never an HTTP
        client library: proxy env vars (`http_proxy`, ...) would leak
        into loopback/tunnel probes on corp-proxy hosts and hang them.
        Every fifth round the server request's state is re-checked so a
        runtime that printed its marker and then crashed fails fast with
        its logs instead of probing a dead endpoint for the whole budget.
        """
        parts = urlsplit(runtime_url)
        host = parts.hostname or "127.0.0.1"
        port = parts.port
        if port is None:
            raise RuntimeError(
                f"runtime URL {runtime_url!r} has no explicit port; check url_template"
            )
        loop = asyncio.get_running_loop()
        rounds = 0
        while loop.time() < deadline:
            if rounds and rounds % 5 == 0:
                state: str | None = None
                try:
                    status = await self._cli.status(server_req)
                    state = str(status.get("state", ""))
                except RuntimeError:
                    state = None  # transient status blip; keep probing
                if state in _TERMINAL_STATES:
                    stdout, stderr = await self._cli.logs(server_req)
                    raise RuntimeError(
                        f"CAPE runtime request {server_req} reached terminal state {state} "
                        f"while waiting for {runtime_url}/health.\n"
                        f"--- workload stdout ---\n{stdout}\n"
                        f"--- workload stderr ---\n{stderr}"
                    )
            rounds += 1
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=2
                )
            except (TimeoutError, OSError):
                await asyncio.sleep(0.5)
                continue
            try:
                writer.write(f"GET /health HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
                await writer.drain()
                status_line = await asyncio.wait_for(reader.readline(), timeout=2)
                if status_line.startswith(b"HTTP/1.") and b" 200 " in status_line:
                    return
            except (TimeoutError, OSError):
                pass
            finally:
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Runtime server not alive at {runtime_url}")

    async def get(self, sandbox_id: SandboxId) -> SandboxInfo:
        record = self._sandboxes.get(sandbox_id)
        if record is None:
            raise KeyError(f"Sandbox not found: {sandbox_id}")
        status = await self._cli.status(record.request_id)
        state = str(status.get("state", ""))
        return SandboxInfo(
            sandbox_id=sandbox_id,
            runtime_url=record.runtime_url,
            status="exited" if state in _TERMINAL_STATES else "running",
        )

    async def delete(self, sandbox_id: SandboxId) -> None:
        """Cancel the sandbox's runtime request.

        Unknown ids are a silent no-op (`session()` calls this on the
        user's exception path, so a raise here would mask the original
        error). Bookkeeping is dropped only after the controller
        confirms the cancel — a rejected/failed cancel keeps the record
        (with a warning) so a later `delete()` can retry instead of
        silently leaking the GPU lease.
        """
        record = self._sandboxes.get(sandbox_id)
        if record is None:
            return
        cancelled = await self._cli.cancel(record.request_id, "agentix_delete")
        if not cancelled:
            logger.warning(
                "delete(%s): cape cancel for request %s was not confirmed; "
                "keeping bookkeeping so delete() can be retried",
                sandbox_id,
                record.request_id,
            )
            return
        self._sandboxes.pop(sandbox_id, None)
        self._inflight_ports.discard(record.runtime_port)
        logger.info("Deleted sandbox %s (request %s)", sandbox_id, record.request_id)


__all__ = ["CapeProvider", "CapeProviderConfig"]
