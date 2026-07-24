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

  - `create()` submits a runtime request whose workload creates the
    per-sandbox workspace, writes an `AGENTIX_ENDPOINT <host> <port>`
    marker file into the per-sandbox meta directory (rw-bound at
    `/agentix-meta`), and execs the bundle's
    `/nix/runtime/bootstrap.sh`. The provider discovers the endpoint by
    polling the *host side* of that marker file, interleaved with
    `cape status` (to fail fast when the workload dies), then
    health-checks `GET /health` on the discovered endpoint with a raw
    TCP probe (never an env-proxy-aware HTTP client, which would hang
    behind a corp proxy or SSH tunnel).
  - `delete()` cancels the runtime request and removes the meta dir;
    bookkeeping is dropped only after the controller confirms the
    cancel (or reports the request as unknown — HTTP 404 — after e.g. a
    journal-less controller restart), so a failed cancel can be retried
    with another `delete()`.
  - `config.bundle` for this backend is an OPAQUE node-visible path to
    an already-extracted bundle tree; it is bind-mounted read-only at
    `/nix`. How the bundle tree gets onto CAPE nodes (shared FS, prior
    upload) is deliberately out of scope — there is no `BundleDeployer`
    in this iteration.
  - `CapeProviderConfig.meta_root` must be a directory visible to both
    the submitter and the pool nodes — the same shared-filesystem
    assumption `bundle` already makes. Each sandbox uses
    `<meta_root>/<sandbox_id>`, created before submit and removed on
    delete.

CLI CONTRACT — verified item by item against the CAPE source. The key
verified facts this provider is built on:

  * `cape run` requires `--pool`, `--user`, and `--task-id` (argparse
    exits 2 without them) and has NO `--workspace` flag. Without
    `--wait`, stdout is exactly one line: the request id (`req-%06d`,
    matched here with `req-\\d{6,}`).
  * `cape status` prints a single JSON object on stdout (the
    human-readable state line goes to stderr). Terminal states are
    COMPLETED / FAILED / CANCELLED / EXPIRED / LOST / INFEASIBLE, and
    the JSON carries no stdout/stderr fields.
  * `cape logs` succeeds for RUNNING requests but returns EMPTY output
    until the command exits — the node agent reports stdout/stderr
    once, after reaping the process; CAPE has no streaming/incremental
    log channel. Endpoint discovery therefore CANNOT go through logs
    (the runtime server never exits); it uses the marker file above.
    Logs are fetched only for terminal requests, as error diagnostics.
  * The token can only be passed as `--token` on argv (the CLI reads no
    env var or file). This provider's token-file/env indirection is
    client-side convenience that feeds that flag; the literal value is
    redacted from error text, but `ps` visibility cannot be avoided
    from the CLI side.

NETWORKING — the client-assigned-port scheme (`AGENTIX_BIND_PORT` plus
a direct probe of `<host>:<port>`) requires the workload to share the
host's network stack. Pin `CapeProviderConfig.runtime_adapter` to a
host-network adapter: apptainer and bubblewrap do not unshare the
network namespace, but CAPE's podman adapter runs without host
networking, so an endpoint bound inside it would be unreachable.
(`local-process` shares everything and suits CPU-only local
validation.) The default stays None — the pool default applies — but
deployments must ensure a host-network adapter serves Agentix requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import shlex
import shutil
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

_REQUEST_ID_RE = re.compile(r"req-\d{6,}")
"""Shape of the request id `cape run` prints on stdout (`req-%06d` in
the CAPE source; more than six digits once the sequence outgrows the
minimum width)."""

_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "LOST", "INFEASIBLE"})
"""Terminal request states in the `cape status` JSON (verified against
the CAPE source; the non-terminal states are SUBMITTED / QUEUED /
LEASED / PREPARING_IMAGE / STARTING_SESSION / RUNNING / RECONCILING)."""

_REASON_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")
"""Characters stripped from `cape cancel --reason` strings."""

_ENDPOINT_MARKER = "AGENTIX_ENDPOINT"
"""Prefix of the marker line the boot script writes to the meta dir."""

_META_MOUNT = "/agentix-meta"
"""In-sandbox mount point of the per-sandbox meta directory (rw bind)."""

_ENDPOINT_FILE_NAME = "endpoint"
"""Marker file name inside the per-sandbox meta directory."""

_BUNDLE_BOOT_RELATIVE = BUNDLE_RUNTIME_ENTRYPOINT.removeprefix(BUNDLE_NIX_ROOT + "/")
"""Bootstrap path relative to the bundle root (`runtime/bootstrap.sh`)."""

_MEMORY_RE = re.compile(r"(\d+)\s*([kmgt])?i?b?", re.IGNORECASE)
_MEMORY_UNIT_BYTES = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}

_REAP_TIMEOUT_SECONDS = 5.0
"""Bounded wait for a killed `cape` subprocess to be reaped."""


class CapeProviderConfig(BaseModel):
    """Submitter-side settings for the CAPE backend.

    Every field is optional or defaulted so `CapeProvider()` constructs
    with zero arguments — the plugin registry instantiates providers
    with `cls()`. Fields the real CLI cannot work without (`pool`,
    `user`, `meta_root`) are checked at first use, not construction.
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
    pool: str | None = Field(
        default=None,
        description="CAPE pool name (`--pool`). The real CLI requires it; "
        "checked at first use.",
    )
    user: str | None = Field(
        default=None,
        description="CAPE user identity (`--user`). The real CLI requires it; "
        "checked at first use.",
    )
    meta_root: str | None = Field(
        default=None,
        description="Directory for per-sandbox meta dirs (`<meta_root>/<sandbox_id>`), "
        "used for endpoint discovery: the boot script writes the endpoint marker "
        "file through a rw bind of the meta dir, and the provider polls the host "
        "side. Must be visible to both the submitter and the pool nodes — the "
        "same shared-filesystem assumption `bundle` makes. Checked at first use.",
    )
    workspace_root: str = Field(
        default="/workspace",
        description="Node-side base directory; each sandbox uses "
        "`<workspace_root>/<sandbox_id>` as its workspace. The boot script "
        "creates it (`mkdir -p`) and cd's into it before exec'ing the runtime — "
        "nothing else creates it, which is also why `--cwd` is never passed "
        "(every CAPE runtime adapter fails at process start on a missing cwd).",
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
        description="Optional `--runtime-adapter`. Production deployments should pin "
        "a host-network adapter (`apptainer` or `bubblewrap`): the client-assigned-"
        "port scheme needs the workload on the host network stack, and CAPE's "
        "podman adapter has no host networking. `local-process` suits CPU-only "
        "local validation.",
    )
    extra_binds: list[str] = Field(
        default_factory=list,
        description="Raw `src:dst[:ro|rw]` bind specs passed through in addition to "
        "the bundle's `/nix` bind and the meta dir's `/agentix-meta` bind.",
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
        description="Consecutive `cape status` failures tolerated during endpoint "
        "discovery before the create is failed (a single controller blip must "
        "not cancel a lease that already queued onto a GPU).",
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


def _required_settings(config: CapeProviderConfig) -> tuple[str, str, str]:
    """Fail fast on the settings the real CLI / discovery cannot work without.

    `cape run` rejects submissions without `--pool`/`--user` (argparse
    exits 2 with a usage error that would otherwise surface as an
    unreadable RuntimeError), and endpoint discovery needs `meta_root`.
    Returns the narrowed `(pool, user, meta_root)` triple.
    """
    pool, user, meta_root = config.pool, config.user, config.meta_root
    if not pool or not user or not meta_root:
        missing = [
            name
            for name, value in (("pool", pool), ("user", user), ("meta_root", meta_root))
            if not value
        ]
        raise RuntimeError(
            "CapeProviderConfig." + ", CapeProviderConfig.".join(missing) + " must be set "
            "before creating sandboxes: the real `cape run` requires --pool and --user, "
            "and endpoint discovery needs meta_root — a directory visible to both the "
            "submitter and the pool nodes (the same shared-filesystem assumption "
            "`bundle` makes)"
        )
    return pool, user, meta_root


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


def _boot_script(*, workspace: str, meta_dir: str, bundle: str) -> str:
    """Workload for the long-lived runtime request.

    Creates and enters the per-sandbox workspace (nothing else creates
    it — `--cwd` is deliberately not passed, because every CAPE runtime
    adapter fails at process start when the cwd does not exist), writes
    the `AGENTIX_ENDPOINT <host> <port>` marker file into the meta dir,
    and execs the bundle's bootstrap entry point.

    The marker goes to `/agentix-meta/endpoint` — the rw bind of the
    host-side per-sandbox meta dir — and the provider polls the host
    side of that file during endpoint discovery. `cape logs` cannot be
    used for discovery: CAPE has no streaming logs (stdout/stderr are
    reported once, after the command exits), so a runtime server that
    never exits never publishes anything through logs.

    Adapters that ignore bind specs but share the host filesystem
    (local-process) fall back to the host-side meta dir path and the
    host-side bundle bootstrap path, so the same script stays honest
    there. A marker-write failure exits non-zero on purpose: the
    request goes terminal and `create()` fails fast with diagnostics
    instead of burning the whole discovery budget.
    """
    quoted_ws = shlex.quote(workspace)
    host_meta = shlex.quote(meta_dir)
    host_boot = shlex.quote(f"{bundle.rstrip('/')}/{_BUNDLE_BOOT_RELATIVE}")
    return (
        f"mkdir -p {quoted_ws} && cd {quoted_ws} || exit 1; "
        f"if [ -d {_META_MOUNT} ]; then m={_META_MOUNT}; else m={host_meta}; fi; "
        f'h=$(hostname -i 2>/dev/null | cut -d" " -f1); [ -n "$h" ] || h=$(hostname); '
        f'printf "{_ENDPOINT_MARKER} %s %s\\n" "$h" "${{{BIND_PORT_ENV}}}" '
        f'> "$m/{_ENDPOINT_FILE_NAME}" || exit 1; '
        f'b={BUNDLE_RUNTIME_ENTRYPOINT}; [ -e "$b" ] || b={host_boot}; '
        f'exec "$b"'
    )


def _parse_endpoint(text: str) -> tuple[str, int] | None:
    """Find the `AGENTIX_ENDPOINT <host> <port>` marker line in `text`.

    The explicit marker prefix keeps unrelated two-token content from
    being misread as an endpoint, and a partially written marker file
    (seen mid-`printf` on a shared FS) simply fails to parse until the
    next poll."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == _ENDPOINT_MARKER and parts[2].isdigit():
            return parts[1], int(parts[2])
    return None


def _read_endpoint_file(path: Path) -> str:
    """Best-effort read of the host-side marker file ('' when absent).

    Synchronous file IO — call via `asyncio.to_thread` (meta_root
    typically lives on a shared FS that must not block the loop)."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _stderr_reports_unknown_request(stderr: str) -> bool:
    """True when `cape` stderr carries the CLI's single-line HTTP-error
    JSON (`{"status": <code>, "error": ...}`) with status 404 — the
    controller does not know the request id (e.g. it restarted without
    a journal), so a cancel can be treated as already done."""
    for line in stderr.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("status") == 404:
            return True
    return False


async def _remove_meta_dir(meta_dir: Path) -> None:
    """Best-effort removal of a per-sandbox meta dir (never raises)."""
    await asyncio.to_thread(shutil.rmtree, meta_dir, ignore_errors=True)


class _CapeCli:
    """The ONE place that knows the `cape` CLI verb surface.

    Every method's contract (verbs, flags, stdout/JSON shapes, terminal
    states) has been verified item by item against the CAPE source.
    The CLI also offers `cape wait`, which this provider deliberately
    does not use: its process exit code is 0 both on timeout (printing
    a non-terminal status) and on null-exit-code terminal states, so it
    must never be trusted — polling `cape status` and parsing the JSON
    `state` field is the only sound approach. Keep all CLI knowledge in
    this class so any future contract correction is a single-class
    change.
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
        task_id: str,
        image: str,
        gpus: int,
        workload: Sequence[str],
        binds: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cpu_cores: int | None = None,
        memory_gb: int | None = None,
    ) -> str:
        """Submit one workload request and return its request id.

        Verified against the real parser:

            cape run --controller-url U --token T --pool P --user USR
                --task-id ID --image IMG --gpus N [--cpu-cores N]
                [--memory-gb N] --isolation-policy X
                [--runtime-adapter Y] --max-duration-seconds T
                --session-key KEY [--bind src:dst[:ro|rw]]...
                [--env K=V]... -- <workload argv>

        `--pool` / `--user` / `--task-id` are required (argparse exits 2
        without them); there is NO `--workspace` flag; `--cwd` is
        deliberately not passed because no component creates the
        directory — the boot script `mkdir -p && cd`'s instead. Without
        `--wait`, stdout (after stripping blank lines) is exactly one
        line matching `req-\\d{6,}`.
        """
        cfg = self._config
        pool, user = cfg.pool, cfg.user
        if not pool or not user:
            raise RuntimeError(
                "`cape run` requires --pool and --user; set CapeProviderConfig.pool "
                "and CapeProviderConfig.user"
            )
        token = await asyncio.to_thread(_read_token, cfg)
        argv: list[str] = ["run", *self._common(token)]
        argv += ["--pool", pool, "--user", user, "--task-id", task_id]
        argv += ["--image", image, "--gpus", str(int(gpus))]
        if cpu_cores is not None:
            argv += ["--cpu-cores", str(int(cpu_cores))]
        if memory_gb is not None:
            argv += ["--memory-gb", str(int(memory_gb))]
        argv += ["--isolation-policy", cfg.isolation_policy]
        if cfg.runtime_adapter:
            argv += ["--runtime-adapter", cfg.runtime_adapter]
        argv += ["--max-duration-seconds", str(int(cfg.max_duration_seconds))]
        argv += ["--session-key", session_key]
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
        """`cape status <req> --controller-url U --token T`.

        Verified: prints one JSON object on stdout (the human-readable
        state line goes to stderr, so it never pollutes the parse) with
        at least `state` and a non-empty `request_id` echo; terminal
        states are COMPLETED / FAILED / CANCELLED / EXPIRED / LOST /
        INFEASIBLE; the JSON carries NO stdout/stderr fields (logs only
        travel through `cape logs`). A `request_id` echo mismatch means
        the CLI answered for a different request and is treated as an
        infrastructure error, not trusted.
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
        """`cape cancel <req> --controller-url U --token T --reason R`.

        Verified: rc=0 confirms the cancel (idempotent no-op on already
        terminal requests, printing the post-cancel status JSON).
        Returns True when the CLI confirmed the cancel OR when the
        controller reported the request as unknown (HTTP 404 in the
        CLI's stderr JSON) — after a journal-less controller restart the
        request is gone and bookkeeping must not stick forever.
        Ordinary failures are swallowed (best-effort) after a redacted
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
            if _stderr_reports_unknown_request(stderr):
                logger.info(
                    "cape cancel %s: controller no longer knows the request (HTTP 404); "
                    "treating it as already gone",
                    request_id,
                )
                return True
            logger.warning(
                "cape cancel %s (reason=%s) exited rc=%d: %s",
                request_id,
                reason,
                rc,
                stderr.strip(),
            )
            return False
        return True

    async def logs(self, request_id: str) -> tuple[str, str]:
        """`cape logs <req> --controller-url U --token T`.

        Verified: prints the workload's stdout/stderr, BUT both are
        empty until the command exits — the node agent reports output
        once, after reaping the process; there is no streaming channel.
        This method is therefore only called on terminal requests (whose
        logs are populated) to enrich error messages. Best-effort:
        failures collapse to `("", "")`.
        """
        try:
            token = await asyncio.to_thread(_read_token, self._config)
            rc, stdout, stderr = await self._exec(
                ["logs", request_id, *self._common(token)], token=token, verb="logs"
            )
        except Exception:
            return "", ""
        if rc != 0:
            return "", ""
        return stdout, stderr


@dataclass
class _CapeSandboxRecord:
    """Bookkeeping for one live sandbox."""

    request_id: str
    session_key: str
    workspace: str
    meta_dir: str
    runtime_url: str
    runtime_port: int


class CapeProvider(SandboxProvider):
    """Sandbox CRUD via the `cape` CLI (see module docstring)."""

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
        _pool, _user, meta_root = _required_settings(self.config)
        sandbox_id = SandboxId(f"cape-{uuid4().hex[:12]}")
        session_key = f"agentix-{sandbox_id}"
        workspace = f"{self.config.workspace_root.rstrip('/')}/{sandbox_id}"
        meta_dir = Path(meta_root).expanduser() / str(sandbox_id)
        port = self._allocate_port()
        try:
            await asyncio.to_thread(meta_dir.mkdir, parents=True, exist_ok=True)
        except OSError as exc:
            self._inflight_ports.discard(port)
            raise RuntimeError(f"cannot create the sandbox meta dir {meta_dir}: {exc}") from exc

        binds = [
            f"{config.bundle}:{BUNDLE_NIX_ROOT}:ro",
            f"{meta_dir}:{_META_MOUNT}:rw",
            *self.config.extra_binds,
        ]
        env = {BIND_PORT_ENV: str(port), **(config.env or {})}
        resource = config.resource
        gpus = resource.gpu if resource is not None and resource.gpu is not None else 0
        boot = _boot_script(workspace=workspace, meta_dir=str(meta_dir), bundle=config.bundle)

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
                task_id=str(sandbox_id),
                image=config.image,
                gpus=gpus,
                workload=["sh", "-c", boot],
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
            await _remove_meta_dir(meta_dir)
            raise
        except BaseException:
            self._inflight_ports.discard(port)
            await _remove_meta_dir(meta_dir)
            raise
        logger.info("CAPE runtime request %s submitted for sandbox %s", server_req, sandbox_id)

        # The runtime request has started: from here every failure or
        # cancellation must best-effort cancel it, drop bookkeeping, and
        # re-raise — `session()` cannot clean up a sandbox it never got.
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.config.create_timeout_seconds
            host, marker_port = await self._discover_endpoint(
                server_req, endpoint_file=meta_dir / _ENDPOINT_FILE_NAME, deadline=deadline
            )
            runtime_url = self.config.url_template.format(host=host, port=marker_port)
            await self._wait_healthy(server_req, runtime_url, deadline)
        except BaseException:
            self._sandboxes.pop(sandbox_id, None)
            self._inflight_ports.discard(port)
            await self._cli.cancel(server_req, "agentix_create_failed")
            await _remove_meta_dir(meta_dir)
            raise

        self._sandboxes[sandbox_id] = _CapeSandboxRecord(
            request_id=server_req,
            session_key=session_key,
            workspace=workspace,
            meta_dir=str(meta_dir),
            runtime_url=runtime_url,
            runtime_port=port,
        )
        logger.info("Created sandbox %s at %s (request %s)", sandbox_id, runtime_url, server_req)
        # No RuntimeClient is instantiated here — `session()` stamps
        # `call_deadline` on the returned handle post-create.
        return Sandbox(sandbox_id=sandbox_id, runtime_url=runtime_url, status="running")

    async def _discover_endpoint(
        self, server_req: str, *, endpoint_file: Path, deadline: float
    ) -> tuple[str, int]:
        """Wait for the workload to write the endpoint marker file.

        CAPE has no streaming logs — `cape logs` returns empty output
        until the command exits, and the runtime server never exits —
        so discovery polls the *host side* of the per-sandbox marker
        file the boot script writes through the rw-bound meta dir,
        interleaved with `cape status` so a workload that died before
        publishing fails fast (its logs ARE populated once the request
        is terminal, and are fetched then for diagnostics). Transient
        status failures are tolerated up to `transient_failure_limit`
        consecutive times so a single controller blip cannot kill a
        create whose lease already queued onto a GPU. No second request
        is ever submitted into the session — the session model is
        serial (RUNNING_COMMAND ↔ CACHED_IDLE), so a same-session probe
        would queue forever behind the never-ending runtime command.
        """
        loop = asyncio.get_running_loop()
        limit = self.config.transient_failure_limit
        failures = 0
        while True:
            marker = await asyncio.to_thread(_read_endpoint_file, endpoint_file)
            if marker:
                endpoint = _parse_endpoint(marker)
                if endpoint is not None:
                    return endpoint
            try:
                status = await self._cli.status(server_req)
            except RuntimeError as exc:
                failures += 1
                if failures >= limit:
                    raise RuntimeError(
                        f"cape status {server_req} failed {failures} consecutive times "
                        f"during endpoint discovery: {exc}"
                    ) from exc
            else:
                failures = 0
                state = str(status.get("state", ""))
                if state in _TERMINAL_STATES:
                    stdout, stderr = await self._cli.logs(server_req)
                    raise RuntimeError(
                        f"CAPE runtime request {server_req} reached terminal state {state} "
                        f"before publishing its endpoint marker.\n"
                        f"--- workload stdout ---\n{stdout}\n"
                        f"--- workload stderr ---\n{stderr}"
                    )
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
        runtime that wrote its marker and then crashed fails fast with
        its logs (populated once terminal) instead of probing a dead
        endpoint for the whole budget.
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
        """Cancel the sandbox's runtime request and remove its meta dir.

        Unknown ids are a silent no-op (`session()` calls this on the
        user's exception path, so a raise here would mask the original
        error). Bookkeeping is dropped only after the controller
        confirms the cancel — or reports the request as unknown (HTTP
        404, e.g. after a journal-less controller restart), which is
        treated as already-gone. A rejected/failed cancel keeps the
        record (with a warning) so a later `delete()` can retry instead
        of silently leaking the GPU lease.
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
        await _remove_meta_dir(Path(record.meta_dir))
        logger.info("Deleted sandbox %s (request %s)", sandbox_id, record.request_id)


__all__ = ["CapeProvider", "CapeProviderConfig"]
